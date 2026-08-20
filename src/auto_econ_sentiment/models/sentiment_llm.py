import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from auto_econ_sentiment.models.sentiment_base import SentimentBase

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "You are an economic sentiment analysis expert. Analyze the sentiment of the following economic text.\n"
    "Respond ONLY with a valid JSON object in this exact format:\n"
    "{\n"
    '  "polarity": <integer: -1 for negative/hawkish, 0 for neutral, 1 for positive/dovish>,\n'
    '  "confidence": <float: between 0.0 and 1.0 indicating certainty>\n'
    "}\n\n"
    "Text:\n{text}"
)



class SentimentLLM(SentimentBase):
    """Sentiment scorer backed by LLM providers (Ollama or OpenAI-compatible).

    Scores texts by extracting polarity in {-1, 0, 1} and confidence in [0, 1]
    from strict JSON responses. Derived continuous sentiment score is computed
    as polarity * confidence.
    """

    def __init__(
        self,
        model_name: str,
        model_name_short: str,
        prompt_template: Optional[str] = None,
        provider: str = "ollama",
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        output_scale: str = "continuous",
        temperature: float = 0.0,
        confidence_cutoff: Optional[float] = None,
        output_schema: Optional[str] = None,
        net_sentiment_formula: str = "positive_minus_negative",
        df_input: Optional[pd.DataFrame] = None,
        text_column: Optional[str] = None,
        id_column: str = "id_text",
        prompt_version: Optional[str] = None,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__(df_input=df_input, text_column=text_column)
        self.model_name = model_name
        self.model_name_short = model_name_short
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.prompt_version = prompt_version or "v1"
        self.provider = str(provider).lower()
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.output_scale = str(output_scale).lower()
        self.temperature = float(temperature)
        self.confidence_cutoff = confidence_cutoff
        self.output_schema = output_schema
        self.net_sentiment_formula = net_sentiment_formula
        self.id_column = id_column

        self.df_labels: Optional[pd.DataFrame] = None
        self.df_sentence_probabilities: Optional[pd.DataFrame] = None
        self.df_sentiment_output: Optional[pd.DataFrame] = None

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        self._validate_config()

    def _validate_config(self) -> None:
        if self.provider not in ("ollama", "openai"):
            raise ValueError(
                f"Unsupported provider '{self.provider}'. Expected 'ollama' or 'openai'."
            )
        if self.output_scale not in ("continuous", "discrete"):
            raise ValueError(
                f"output_scale must be 'continuous' or 'discrete', got '{self.output_scale}'."
            )
        if self.net_sentiment_formula not in ("positive_minus_negative", "negative_minus_positive"):
            raise ValueError(
                "net_sentiment_formula must be either 'positive_minus_negative' "
                "or 'negative_minus_positive'."
            )

    def format_prompt(self, text: str) -> str:
        """Format the prompt template with the input text."""
        if "{text}" in self.prompt_template:
            return self.prompt_template.replace("{text}", text)
        return f"{self.prompt_template}\n\n{text}"


    def _get_ollama_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        env_host = os.environ.get("API_OLLAMA")
        if env_host:
            return env_host.rstrip("/")
        return "http://localhost:11434"

    def _get_openai_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        return "https://api.openai.com/v1"

    def _get_openai_api_key(self) -> str:
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if not key:
                self.logger.warning(
                    "API key environment variable '%s' is not set or empty.", self.api_key_env
                )
            return key
        return os.environ.get("OPENAI_API_KEY", "")

    def _build_ollama_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self._get_ollama_base_url()}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
            "format": "json",
        }
        return url, headers, payload

    def _build_openai_request(self, prompt: str) -> tuple[str, dict[str, str], dict[str, Any]]:
        base_url = self._get_openai_base_url()
        if base_url.endswith("/chat/completions"):
            url = base_url
        elif base_url.endswith("/v1"):
            url = f"{base_url}/chat/completions"
        else:
            url = f"{base_url}/v1/chat/completions" if "/v1" not in base_url else f"{base_url}/chat/completions"

        api_key = self._get_openai_api_key()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        return url, headers, payload

    def _call_provider(self, prompt: str) -> str:
        """Call the configured LLM provider and return the raw response string."""
        if self.provider == "ollama":
            url, headers, payload = self._build_ollama_request(prompt)
            req = urllib.request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            return str(resp_data.get("response", ""))
        elif self.provider == "openai":
            url, headers, payload = self._build_openai_request(prompt)
            req = urllib.request.Request(
                url=url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
            choices = resp_data.get("choices", [])
            if choices and "message" in choices[0]:
                return str(choices[0]["message"].get("content", ""))
            return ""
        else:
            raise ValueError(f"Unknown provider '{self.provider}'")

    def _extract_json_dict(self, raw_text: str) -> Optional[dict[str, Any]]:
        """Extract a dictionary from raw text using json.loads or regex fallbacks."""
        if not raw_text or not isinstance(raw_text, str):
            return None
        text_str = raw_text.strip()
        try:
            data = json.loads(text_str)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        # Regex fallback 1: code block ```json ... ```
        code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text_str)
        if code_block:
            try:
                data = json.loads(code_block.group(1))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        # Regex fallback 2: first JSON object {...}
        brace_match = re.search(r"\{[\s\S]*?\}", text_str)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _parse_response(self, raw_text: str) -> tuple[Optional[int], Optional[float]]:
        """Parse polarity and confidence from LLM response.

        Returns (polarity, confidence) on success, or (None, None) on failure.
        Logs warnings on unparseable or out-of-range responses without raising.
        """
        try:
            data = self._extract_json_dict(raw_text)
            if data is None:
                self.logger.warning("Failed to parse JSON from LLM response: %r", raw_text)
                return None, None

            if "polarity" not in data or "confidence" not in data:
                self.logger.warning(
                    "Missing 'polarity' or 'confidence' in response: %s", data
                )
                return None, None

            raw_pol = data["polarity"]
            raw_conf = data["confidence"]

            # Validate polarity in {-1, 0, 1}
            try:
                pol_float = float(raw_pol)
                pol_int = int(round(pol_float))
                if pol_float != pol_int or pol_int not in (-1, 0, 1):
                    self.logger.warning("Polarity %r not in {-1, 0, 1}.", raw_pol)
                    return None, None
                polarity = pol_int
            except (ValueError, TypeError):
                self.logger.warning("Invalid polarity value: %r", raw_pol)
                return None, None

            # Validate confidence in [0, 1]
            try:
                conf_float = float(raw_conf)
                if not (0.0 <= conf_float <= 1.0):
                    self.logger.warning("Confidence %r not in range [0, 1].", raw_conf)
                    return None, None
                confidence = conf_float
            except (ValueError, TypeError):
                self.logger.warning("Invalid confidence value: %r", raw_conf)
                return None, None

            return polarity, confidence

        except Exception as exc:
            self.logger.warning("Unexpected error parsing response %r: %s", raw_text, exc)
            return None, None

    def _calculate_score(
        self,
        polarity: Optional[int],
        confidence: Optional[float],
        confidence_cutoff: Optional[float] = None,
    ) -> float:
        """Derive score from polarity and confidence according to output_scale."""
        if polarity is None or confidence is None:
            return np.nan

        cutoff = confidence_cutoff if confidence_cutoff is not None else self.confidence_cutoff
        if cutoff is not None and confidence < cutoff:
            return np.nan

        if self.output_scale == "continuous":
            return float(polarity) * float(confidence)
        elif self.output_scale == "discrete":
            # Map -1 -> 0, 0 -> 1, 1 -> 2
            return float(polarity + 1)
        else:
            return float(polarity) * float(confidence)

    def analyze_sentiment_single(
        self,
        texts: Union[pd.Series, list[str]],
    ) -> pd.DataFrame:
        """Run LLM inference over texts and return scoring DataFrame."""
        if isinstance(texts, pd.Series):
            text_list = texts.fillna("").astype(str).tolist()
        else:
            text_list = ["" if t is None else str(t) for t in texts]

        polarities: list[float] = []
        confidences: list[float] = []
        scores: list[float] = []

        prefix = self.model_name_short

        for text in tqdm(text_list, desc=f"LLM Sentiment ({prefix})"):
            if not text.strip():
                polarities.append(np.nan)
                confidences.append(np.nan)
                scores.append(np.nan)
                continue

            prompt = self.format_prompt(text)
            try:
                raw_resp = self._call_provider(prompt)
            except Exception as exc:
                self.logger.warning("Error calling provider for text %r: %s", text[:50], exc)
                raw_resp = ""

            pol, conf = self._parse_response(raw_resp)
            polarities.append(float(pol) if pol is not None else np.nan)
            confidences.append(float(conf) if conf is not None else np.nan)
            score = self._calculate_score(pol, conf)
            scores.append(score)

        result_df = pd.DataFrame(
            {
                f"{prefix}_polarity": polarities,
                f"{prefix}_confidence": confidences,
                f"{prefix}_sentiment_byalltext": scores,
                f"{prefix}_provider": self.provider,
                f"{prefix}_model": self.model_name,
                f"{prefix}_prompt_version": self.prompt_version,
                f"{prefix}_temperature": self.temperature,
            }
        )
        return result_df

    def analyze_sentiment(self) -> pd.DataFrame:
        """Analyze sentiment for ``self.input_df`` and store results."""
        if self.input_df is None or self.text_column is None:
            raise ValueError("Input DataFrame and text column must be set before analysis.")
        if self.text_column not in self.input_df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in input DataFrame.")

        predictions = self.analyze_sentiment_single(self.input_df[self.text_column])
        self.df_labels = pd.concat([self.input_df.reset_index(drop=True), predictions], axis=1)
        return self.df_labels

    def sentiment_bysentence(
        self,
        confidence_cutoff: Optional[float] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate sentence-level classifications back to ``id_text``."""
        if self.df_labels is None:
            self.analyze_sentiment()
        assert self.df_labels is not None

        if self.id_column not in self.df_labels.columns:
            raise ValueError(f"Sentence-level aggregation requires an '{self.id_column}' column.")

        cutoff = (
            confidence_cutoff
            if confidence_cutoff is not None
            else (self.confidence_cutoff if self.confidence_cutoff is not None else 0.7)
        )

        prefix = self.model_name_short
        pol_col = f"{prefix}_polarity"
        conf_col = f"{prefix}_confidence"

        df_sentence_details = self.df_labels.set_index(self.id_column)[[pol_col, conf_col]].copy()

        valid_mask = self.df_labels[pol_col].notna() & self.df_labels[conf_col].ge(cutoff)

        pos_mask = valid_mask & (self.df_labels[pol_col] == 1)
        neg_mask = valid_mask & (self.df_labels[pol_col] == -1)
        neu_mask = valid_mask & (self.df_labels[pol_col] == 0)

        df_working = self.df_labels[[self.id_column]].copy()
        df_working["_pos"] = pos_mask.astype(int)
        df_working["_neg"] = neg_mask.astype(int)
        df_working["_neu"] = neu_mask.astype(int)

        agg = df_working.groupby(self.id_column).sum()

        df_score = pd.DataFrame(index=agg.index)
        df_score[f"{prefix}_count_positive"] = agg["_pos"]
        df_score[f"{prefix}_count_neutral"] = agg["_neu"]
        df_score[f"{prefix}_count_negative"] = agg["_neg"]
        df_score[f"{prefix}_countsentence_positive"] = agg["_pos"]
        df_score[f"{prefix}_countsentence_neutral"] = agg["_neu"]
        df_score[f"{prefix}_countsentence_negative"] = agg["_neg"]

        denominator = (agg["_pos"] + agg["_neu"] + agg["_neg"]).replace(0, np.nan)
        df_score[f"{prefix}_share_positive"] = (agg["_pos"] / denominator).fillna(0.0)
        df_score[f"{prefix}_share_neutral"] = (agg["_neu"] / denominator).fillna(0.0)
        df_score[f"{prefix}_share_negative"] = (agg["_neg"] / denominator).fillna(0.0)

        pos_share = df_score[f"{prefix}_share_positive"]
        neg_share = df_score[f"{prefix}_share_negative"]

        if self.net_sentiment_formula == "negative_minus_positive":
            df_score[f"{prefix}_net_sentiment"] = neg_share - pos_share
        elif self.net_sentiment_formula == "positive_minus_negative":
            df_score[f"{prefix}_net_sentiment"] = pos_share - neg_share
        else:
            raise ValueError(
                "net_sentiment_formula must be either 'positive_minus_negative' "
                "or 'negative_minus_positive'."
            )

        df_score[f"{prefix}_sentiment_bysentence"] = df_score[f"{prefix}_net_sentiment"]

        return df_score, df_sentence_details

    def sentiment_pipeline(
        self,
        aggregation: str = "byalltext",
        confidence_cutoff: Optional[float] = None,
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
        """Run LLM sentiment pipeline and assign ``self.df_final``."""
        aggregation = aggregation.lower()
        self.analyze_sentiment()
        if aggregation == "byalltext":
            assert self.df_labels is not None
            self.df_sentiment_output = self.df_labels
            self.df_final = self.df_sentiment_output
            return self.df_sentiment_output
        elif aggregation == "bysentence":
            df_agg, df_sentence_details = self.sentiment_bysentence(
                confidence_cutoff=confidence_cutoff
            )
            self.df_sentiment_output = df_agg
            self.df_sentence_probabilities = df_sentence_details
            self.df_final = self.df_sentiment_output
            return df_agg, df_sentence_details
        else:
            raise ValueError("aggregation must be either 'byalltext' or 'bysentence'.")
