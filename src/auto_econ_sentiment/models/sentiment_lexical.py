import yaml
import pandas as pd
from typing import Optional
from sklearn.feature_extraction.text import CountVectorizer
from auto_econ_sentiment.models.sentiment_base import SentimentBase
import os
import logging

class SentimentLexical(SentimentBase):
    """Lexical sentiment scorer that counts dictionary hits in a text column and
    aggregates them into a per-document score.

    Supported aggregations: ``"posneg"`` (positive vs. negative tone) and
    ``"allwords"`` (positive minus negative, normalized by total tokens).
    """

    def __init__(
        self,
        df_input: pd.DataFrame,
        text_column: str = 'text',
        dictionary_path: Optional[str] = None,
        log_level: int = logging.WARNING,
    ) -> None:
        super().__init__(df_input, text_column)
        if dictionary_path is None:
            dictionary_path = os.path.join(os.path.dirname(__file__), "..", "data", "lexical_master_dict.yaml")
        self.dictionary_path = dictionary_path
        self.bow = None
        self.words_pos = None
        self.words_neg = None
        self.df_final = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"SentimentLexical initialized with {len(df_input)} records, text column: '{text_column}'")

    def _prepare_dictionary(self, dictionary_name: str):
        self.logger.info(f"Loading dictionary: {dictionary_name} from {self.dictionary_path}")
        try:
            with open(self.dictionary_path, 'r') as f:
                all_dicts = yaml.safe_load(f)
            if dictionary_name not in all_dicts:
                raise KeyError(f"Dictionary '{dictionary_name}' not found in {self.dictionary_path}")
            dict_data = all_dicts[dictionary_name]
            self.words_pos = [str(w) for w in dict_data.get('positive', [])]
            self.words_neg = [str(w) for w in dict_data.get('negative', [])]
            vocab = self.words_pos + self.words_neg + [str(w) for w in dict_data.get('neutral', [])]
            self.bow = CountVectorizer(vocabulary=vocab)
            self.logger.info(f"Dictionary loaded: {len(vocab)} total words, {len(self.words_pos)} positive, {len(self.words_neg)} negative")
        except Exception as e:
            self.logger.error(f"Error loading dictionary: {e}")
            raise

    def _process_vectorize_text(self, text_series: pd.Series) -> pd.DataFrame:
        self.logger.info(f"Vectorizing {len(text_series)} text records")
        try:
            count_array = self.bow.fit_transform(text_series).toarray()
            df_bow = pd.DataFrame(data=count_array, columns=self.bow.get_feature_names_out(), index=text_series.index)
            self.logger.debug(f"Vectorization complete, shape: {df_bow.shape}")
            return df_bow
        except Exception as e:
            self.logger.error(f"Error in vectorization: {e}")
            raise

    def _process_count_sentiment_words(self, df_bow: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Counting sentiment words")
        try:
            pos_counts = df_bow.loc[:, df_bow.columns.isin(self.words_pos)]
            neg_counts = df_bow.loc[:, df_bow.columns.isin(self.words_neg)]
            pos_counts_sum, neg_counts_sum = pos_counts.sum(axis=1), neg_counts.sum(axis=1)
            
            # Create dictionaries for words with count > 0
            words_positive = pos_counts.apply(
                lambda row: {word: count for word, count in row.items() if count > 0}, 
                axis=1
            )
            words_negative = neg_counts.apply(
                lambda row: {word: count for word, count in row.items() if count > 0}, 
                axis=1
            )
            
            result_df = pd.concat([
                pos_counts_sum.to_frame().rename(columns={0: 'counttoken_positive'}),
                neg_counts_sum.to_frame().rename(columns={0: 'counttoken_negative'}),
                words_positive.to_frame().rename(columns={0: 'words_positive'}),
                words_negative.to_frame().rename(columns={0: 'words_negative'})
            ], axis=1).fillna(0)
            
            self.df_bow = df_bow.copy()
            self.logger.debug(f"Sentiment word counts: avg positive={result_df['counttoken_positive'].mean():.2f}, avg negative={result_df['counttoken_negative'].mean():.2f}")
            return result_df
        except Exception as e:
            self.logger.error(f"Error in counting sentiment words: {e}")
            raise

    def _postprocess_aggregate_sentiment(self, df_count: pd.DataFrame, method: str) -> pd.DataFrame:
        self.logger.info(f"Aggregating sentiment using method: {method}")
        try:
            if method == 'posneg':
                df_count['sentiment'] = df_count.apply(
                    lambda x: 1 + ((x['counttoken_positive'] - x['counttoken_negative']) / (x['counttoken_positive'] + x['counttoken_negative'])) 
                    if (x['counttoken_positive'] + x['counttoken_negative']) > 0 else 1,  # Avoid division by zero
                    axis=1
                )
                self.logger.debug(f"Sentiment scores generated using 'posneg' method, mean: {df_count['sentiment'].mean():.2f}")
            
            elif method == 'allwords':
                # Calculate total tokens for each document
                total_tokens = CountVectorizer(stop_words='english').fit_transform(self.input_df[self.text_column]).toarray().sum(axis=1)
                # Add total tokens to the dataframe for reference
                df_count['counttoken_total'] = total_tokens
                
                # Calculate sentiment using total tokens in denominator
                df_count['sentiment'] = df_count.apply(
                    lambda x: 1 + ((x['counttoken_positive'] - x['counttoken_negative']) / x['counttoken_total']) 
                    if x['counttoken_total'] > 0 else 1,  # Avoid division by zero
                    axis=1
                )
                self.logger.debug(f"Sentiment scores generated using 'allwords' method, mean: {df_count['sentiment'].mean():.2f}")
            
            else:
                self.logger.warning(f"Unknown aggregation method: {method}, no sentiment score calculated")
            
            df_count = df_count.add_suffix(f"_{method}")
            return df_count
        except Exception as e:
            self.logger.error(f"Error in sentiment aggregation: {e}")
            raise

    def sentiment_pipeline(
        self,
        dictionary_name: str,
        method: str = 'posneg',
        text_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Score ``text_column`` (or the configured default) against the named
        dictionary and return the input DataFrame joined with the sentiment
        columns, prefixed by ``dictionary_name`` and suffixed by ``method``."""
        active_text_column = text_column if text_column is not None else self.text_column
        self.logger.info(f"Starting sentiment pipeline with dictionary '{dictionary_name}' and method '{method}'")
        try:
            self._prepare_dictionary(dictionary_name)
            df_bow = self._process_vectorize_text(self.input_df[active_text_column])
            df_counts_sentiment = self._process_count_sentiment_words(df_bow)
            df_final = self._postprocess_aggregate_sentiment(df_counts_sentiment, method).add_prefix(f"{dictionary_name}_")
            self.df_final = pd.concat([self.input_df, df_final], axis=1)
            self.logger.info(f"Sentiment pipeline completed successfully, results shape: {self.df_final.shape}")
            return self.df_final
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

    def export_results(self, export_path: str, index: bool = False) -> None:
        """Write the most recent ``sentiment_pipeline`` result to ``export_path`` as CSV."""
        self.logger.info(f"Exporting results to {export_path}")
        try:
            if self.df_final is None:
                self.logger.error("Cannot export: No results available, run sentiment_pipeline first")
                return
                
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            self.df_final.to_csv(export_path, index=index)
            self.logger.info(f"Results exported successfully to {export_path}")
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            raise
