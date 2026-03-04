from tqdm import tqdm
import re
import os
from pathlib import Path
from html.parser import HTMLParser
import pandas as pd
import spacy
import unicodedata
import html
import logging

try: 
    import nltk
    nltk.data.find('tokenizers/punkt_tab/english/')
except (LookupError, ImportError):
    try:
        import nltk
        nltk.download('punkt_tab')
    except ImportError:
        pass
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize    

class _HTMLStripper(HTMLParser):
    _SKIP_TAGS = {"script", "style", "head", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._buf = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._SKIP_TAGS:
            self._skip = True

    def handle_endtag(self, tag):
        if tag.lower() in self._SKIP_TAGS:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def get_text(self):
        return " ".join(self._buf)


class TextCleaner:
    def __init__(self, df, text_column, clean_config=None, export_path=None):
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        self.df = df.copy()
        self.df['id_text'] = range(1, df.shape[0] + 1)
        self.text_column = text_column
        self.export_path = export_path
        self.clean_config = {
            'clean_html': True,
            'clean_numbers_percentages': True,
            'normalize_unicode': True,
            'fix_encoding': True,
            'normalize_whitespace': True,
            'remove_extra_punctuation': True,
            'standardize_quotes': True,
            'header_patterns': [],
            'footer_patterns': [],
            'remove_headers': [],
            'min_sentence_length': 10,
            'max_sentence_length': 1000,
            'tokenize': False,
            'stem': False,
        }
        self.stemmer = PorterStemmer()
        if clean_config:
            self.clean_config.update(clean_config)

        if 'remove_headers' in self.clean_config and self.clean_config['remove_headers']:
            if not self.clean_config.get('header_patterns'):
                self.clean_config['header_patterns'] = self.clean_config['remove_headers']

        self._load_spacy_model()

    def _load_spacy_model(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("spaCy model not found. Falling back to NLTK.")
            self.nlp = None

    @staticmethod
    def clean_html(text):
        stripper = _HTMLStripper()
        stripper.feed(text)
        return stripper.get_text()

    @staticmethod
    def clean_spaces_html(text):
        logging.debug("Cleaning HTML and spaces")
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", " ", text)
        text = text.strip()
        return text

    @staticmethod
    def normalize_unicode(text):
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[^\x00-\x7F]+', lambda m: unicodedata.normalize('NFKD', m.group()).encode('ascii', 'ignore').decode('ascii'), text)
        return text

    @staticmethod
    def fix_encoding_issues(text):
        replacements = {
            'â€™': "'", 'â€œ': '"', 'â€\x9d': '"', 'â€"': '–', 'â€"': '—',
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã ': 'à', 'Ã¨': 'è', 'Ã¬': 'ì', 'Ã²': 'ò', 'Ã¹': 'ù',
            'Ã±': 'ñ', 'Ã§': 'ç', 'Â': '',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def normalize_whitespace(text):
        text = re.sub(r'\r\n|\r|\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        return text.strip()

    @staticmethod
    def standardize_quotes(text):
        text = re.sub(r'[''`]', "'", text)
        text = re.sub(r'[""„"]', '"', text)
        return text

    @staticmethod
    def clean_numbers_percentages(text):
        logging.debug("Cleaning numbers and percentages")
        replacements = {
            '¼': '.25', '½': '.5', '¾': '.75',
            '⅓': '.33', '⅔': '.67', '⅛': '.125',
            '⅜': '.375', '⅝': '.625', '⅞': '.875',
            '1/2': '.5', '3/4': '.75', '1/4': '.25',
            '1/3': '.33', '2/3': '.67',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        text = re.sub(r'percentage', '%', text)
        text = re.sub(r'percent', '%', text)
        text = re.sub(r'per cent', '%', text)
        text = re.sub(r'(\d+)\s*%', r'\1%', text)
        return text.strip()

    @staticmethod
    def remove_extra_punctuation(text):
        text = re.sub(r'[.]{2,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'[;]{2,}', ';', text)
        text = re.sub(r'[-]{3,}', '---', text)
        return text

    def remove_header(self, text, patterns=None):
        try:
            logging.debug(f"Removing headers with patterns: {patterns}")
            if patterns and isinstance(patterns, list) and len(patterns) > 0:
                for pattern in patterns:
                    escaped_pattern = re.escape(pattern)
                    if re.match(f'^{escaped_pattern}', text, re.IGNORECASE | re.DOTALL):
                        text = re.sub(f'^{escaped_pattern}', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
                        logging.debug(f"Removed header pattern: {pattern}")
                        break
            return text
        except Exception as e:
            logging.error(f"Error in remove_header: {e}")
            return text

    def remove_footer(self, text, patterns=None):
        try:
            logging.debug(f"Removing footers with patterns: {patterns}")
            if patterns and isinstance(patterns, list) and len(patterns) > 0:
                for pattern in patterns:
                    escaped_pattern = re.escape(pattern)
                    if re.search(f'{escaped_pattern}$', text, re.IGNORECASE | re.DOTALL):
                        text = re.sub(f'{escaped_pattern}$', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
                        logging.debug(f"Removed footer pattern: {pattern}")
                        break
            return text
        except Exception as e:
            logging.error(f"Error in remove_footer: {e}")
            return text
    
    # New methods to add to the class
    def tokenize_text(self, text):
        try:
            if not isinstance(text, str) or not text.strip():
                return []
            
            try:
                tokens = word_tokenize(text.lower())
            except ImportError:
                tokens = re.findall(r'\b\w+\b', text.lower())
            
            tokens = [token for token in tokens if token.isalpha()]
            return tokens
        except Exception as e:
            logging.error(f"Error in tokenize_text: {e}")
            return []

    def stem_tokens(self, tokens):
        try:
            if not tokens or not hasattr(self, 'stemmer'):
                return tokens
            stems = [self.stemmer.stem(token) for token in tokens]
            stems = " ".join(stems)
            return stems
        except Exception as e:
            logging.error(f"Error in stem_tokens: {e}")
            return tokens

    def clean_text(self, text, clean_spaces_html=True, clean_numbers_percentages=True):
        try:
            if not isinstance(text, str):
                text = str(text)
            
            if not text.strip():
                return ""

            logging.debug("Starting text cleaning")
            original_length = len(text)

            if self.clean_config.get('fix_encoding', True):
                text = self.fix_encoding_issues(text)

            if self.clean_config.get('normalize_unicode', True):
                text = self.normalize_unicode(text)

            if clean_spaces_html and self.clean_config.get('clean_html', True):
                text = self.clean_spaces_html(text)

            if self.clean_config.get('standardize_quotes', True):
                text = self.standardize_quotes(text)

            if clean_numbers_percentages and self.clean_config.get('clean_numbers_percentages', True):
                text = self.clean_numbers_percentages(text)

            if self.clean_config.get('remove_extra_punctuation', True):
                text = self.remove_extra_punctuation(text)

            if self.clean_config.get('normalize_whitespace', True):
                text = self.normalize_whitespace(text)

            header_patterns = self.clean_config.get('header_patterns', [])
            if len(header_patterns) > 0:
                logging.debug(f"Applying header removal with {len(header_patterns)} patterns")
                text = self.remove_header(text, patterns=header_patterns)
            
            footer_patterns = self.clean_config.get('footer_patterns', [])
            if len(footer_patterns) > 0:
                logging.debug(f"Applying footer removal with {len(footer_patterns)} patterns")
                text = self.remove_footer(text, patterns=footer_patterns)

            logging.debug(f"Text cleaning complete. Length: {original_length} -> {len(text)}")
            return text.strip()
            
        except Exception as e:
            logging.error(f"Error in clean_text: {e}")
            return str(text)

    def _sentence_tokenize(self, text):
        try:
            if not text.strip():
                return []

            if self.nlp:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
            else:
                try:
                    from nltk.tokenize import sent_tokenize
                    sentences = sent_tokenize(text)
                except ImportError:
                    sentences = self._fallback_sentence_split(text)

            min_len = self.clean_config.get('min_sentence_length', 10)
            max_len = self.clean_config.get('max_sentence_length', 1000)
            
            filtered_sentences = []
            for sent in sentences:
                sent = re.sub(r"\n+", "", sent).strip()
                if min_len <= len(sent) <= max_len:
                    sent = re.sub(r'\s+', ' ', sent)
                    if sent and not re.match(r'^[^\w]*$', sent):
                        filtered_sentences.append(sent)
            
            return filtered_sentences
        except Exception as e:
            logging.error(f"Error in sentence tokenization: {e}")
            return [text]

    def _fallback_sentence_split(self, text):
        abbreviations = {
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.',
            'etc.', 'e.g.', 'i.e.', 'cf.', 'al.', 'Inc.', 'Ltd.', 'Corp.',
            'Co.', 'U.S.', 'U.K.', 'E.U.', 'N.Y.', 'L.A.', 'D.C.',
            'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.', 'Jul.', 'Aug.',
            'Sep.', 'Oct.', 'Nov.', 'Dec.'
        }
        
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace('.', '<PERIOD>'))
        
        sentences = re.split(r'[.!?]+\s+', text)
        
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.replace('<PERIOD>', '.')
            if sentence.strip():
                processed_sentences.append(sentence.strip())
        
        return processed_sentences

    def process_text(self):
        try:
            logging.info("Starting text processing")
            logging.info(f"Processing {len(self.df)} rows")
            
            self.df['text_clean'] = self.df[self.text_column].apply(
                lambda x: self.clean_text(
                    str(x),
                    clean_spaces_html=self.clean_config.get('clean_html', True),
                    clean_numbers_percentages=self.clean_config.get('clean_numbers_percentages', True)
                )
            )
            logging.info("Text cleaning completed")
            
            # Tokenization
            if self.clean_config.get('tokenize', False):
                logging.info("Starting tokenization")
                try:
                    tqdm.pandas(desc="Tokenizing text", position=0, leave=True)
                    self.df['text_tokens'] = self.df['text_clean'].progress_apply(self.tokenize_text)
                    self.df['text_tokens_str'] = self.df['text_tokens'].apply(lambda token_list: " ".join(token_list))
                except ImportError:
                    self.df['text_tokens'] = self.df['text_clean'].apply(self.tokenize_text)
                logging.info("Tokenization completed")

            # Stemming
            if self.clean_config.get('stem', False):
                logging.info("Starting stemming")
                try:
                    tqdm.pandas(desc="Stemming tokens", position=0, leave=True)
                    self.df['text_stems'] = self.df['text_tokens'].progress_apply(self.stem_tokens)
                except ImportError:
                    self.df['text_stems'] = self.df['text_tokens'].apply(self.stem_tokens)
                logging.info("Stemming completed")
            
            logging.info("Starting sentence tokenization")
            try:
                tqdm.pandas(desc="Tokenizing sentences", position=0, leave=True)
                self.df['text_sentences'] = self.df['text_clean'].progress_apply(self._sentence_tokenize)
            except ImportError:
                self.df['text_sentences'] = self.df['text_clean'].apply(self._sentence_tokenize)
            logging.info("Sentence tokenization completed")
            
            return self.df
            
        except Exception as e:
            logging.error(f"Error in process_text: {e}")
            raise

    def process_sentences(self):
        try:
            logging.info("Processing sentences")
            self.df = self.df.assign(text_sentence=self.df['text_sentences'])
            df_sentences = self.df.explode('text_sentence')
            df_sentences['id_sentence'] = df_sentences.groupby(['id_text']).cumcount() + 1
            df_sentences['id_textsent'] = df_sentences.apply(lambda col: f"{col['id_text']}_{col['id_sentence']}", axis=1)
            df_sentences['len_sent'] = df_sentences.text_sentence.apply(len)
            self.df_sentences = df_sentences.copy().drop(['Unnamed: 0'], errors='ignore', axis=1)
            logging.info("Sentence processing completed")
        except Exception as e:
            logging.error(f"Error in process_sentences: {e}")
            raise

    def export_data(self):
        try:
            if self.export_path:
                logging.info(f"Exporting data to {self.export_path}")
                output_dir = Path(self.export_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                self.df.to_parquet(output_dir / 'cleaned.parquet.gzip', compression='gzip', index=False)
                self.df_sentences.drop(['text','text_sentences', 'text_clean', 'event','title','speaker','text_len','text_tokens', 'text_stems'], axis=1, errors='ignore').to_parquet(output_dir / 'cleaned_sentences.parquet.gzip', compression='gzip', index=False)
                return (output_dir / 'cleaned.parquet.gzip', output_dir / 'cleaned_sentences.parquet.gzip',)
            else:
                logging.info("Export path not specified. Data will not be exported.")
                return None
        except Exception as e:
            logging.error(f"Error in export_data: {e}")
            raise

    def run(self, clean_config=None):
        try:
            logging.info("Starting TextCleaner.run()")
            if clean_config:
                self.clean_config.update(clean_config)
                if 'remove_headers' in clean_config and clean_config['remove_headers']:
                    if not self.clean_config.get('header_patterns'):
                        self.clean_config['header_patterns'] = clean_config['remove_headers']

            logging.info("Running process_text()")
            df_text = self.process_text()

            logging.info("Running process_sentences()")
            self.process_sentences()

            if self.export_path:
                logging.info("Running export_data()")
                export_paths = self.export_data()
                if export_paths:
                    logging.info(f"Data exported to: {export_paths}")

            logging.info("TextCleaner.run() completed successfully")
            return df_text
            
        except Exception as e:
            logging.error(f"Error in TextCleaner.run(): {e}")
            raise
