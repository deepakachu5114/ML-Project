import constants
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from string import punctuation
import re
from logger_config import logger

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")


class PreprocessForNonContextualEmbeddings:
    def __init__(self, data_path=constants.DATAPATH):
        self.train = pd.read_csv(f"{data_path}/dreaddit-train.csv")
        self.test = pd.read_csv(f"{data_path}/dreaddit-test.csv")
        self.stop_words = set(stopwords.words('english'))
        logger.info("Preprocessing for non-contextual embeddings initialized.")

    def _preprocess(self, text):
        """
        Run the complete preprocessing pipeline:
        remove punctuation, remove stopwords, tokenize, and lemmatize.
        """
        try:
            text = self._removepunctuation(text)
            tokens = self._tokenize(text)
            tokens = self._removestopwords(tokens)
            lemmatized_tokens = self._lemmatize(tokens)
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None
        return " ".join(lemmatized_tokens)

    def _removestopwords(self, tokens):
        """
        Remove stopwords from a list of tokens.
        """
        return [word for word in tokens if word.lower() not in self.stop_words]

    def _removepunctuation(self, text):
        """
        Remove punctuation from text.
        """
        return ''.join([char for char in text if char not in punctuation])

    def _lemmatize(self, tokens):
        """
        Lemmatize the tokens using SpaCy.
        """
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]

    def _tokenize(self, text):
        """
        Tokenize the text using NLTK.
        """
        return nltk.word_tokenize(text)

    def _filter_columns(self, df):
        """
        Keep only 'subreddit', 'text', 'label', and 'confidence' columns.
        """
        return df[['subreddit', 'text', 'label', 'confidence']]

    def save(self, savepath=constants.SAVEPATH):
        """
        Apply preprocessing to only the 'text' column of the train and test data,
        and save the cleaned data to new files, while keeping only specified columns.
        """
        # Keep only the required columns
        self.train = self._filter_columns(self.train)
        self.test = self._filter_columns(self.test)

        # Apply preprocessing only to the 'text' column
        self.train['text'] = self.train['text'].apply(self._preprocess)
        self.test['text'] = self.test['text'].apply(self._preprocess)

        # Save the new processed datasets
        self.train.to_csv(f"{savepath}/train_noncontextual_preprocessed.csv", index=False)
        self.test.to_csv(f"{savepath}/test_noncontextual_preprocessed.csv", index=False)

        logger.info("Preprocessing for non-contextual embeddings completed.")


class PreprocessForContextualEmbeddings:
    def __init__(self, data_path=constants.DATAPATH):
        self.train = pd.read_csv(f"{data_path}/dreaddit-train.csv")
        self.test = pd.read_csv(f"{data_path}/dreaddit-test.csv")
        logger.info("Preprocessing for contextual embeddings initialized.")

    def _preprocess(self, text):
        """
        Run basic preprocessing for contextual embeddings.
        Removes URLs, punctuation, and lowercases the text.
        """
        try:
            text = self._remove_urls(text)
            text = text.lower()
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return None
        return text

    def _remove_urls(self, text):
        """
        Remove URLs from the text using regex.
        """
        url_pattern = r'http[s]?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)


    def _filter_columns(self, df):
        """
        Keep only 'subreddit', 'text', 'label', and 'confidence' columns.
        """
        return df[['subreddit', 'text', 'label', 'confidence']]

    def save(self, savepath=constants.SAVEPATH):
        """
        Apply preprocessing to only the 'text' column of the train and test data,
        and save the cleaned data to new files, while keeping only specified columns.
        """
        # Keep only the required columns
        self.train = self._filter_columns(self.train)
        self.test = self._filter_columns(self.test)

        # Apply preprocessing only to the 'text' column
        self.train['text'] = self.train['text'].apply(self._preprocess)
        self.test['text'] = self.test['text'].apply(self._preprocess)

        # Save the new processed datasets
        self.train.to_csv(f"{savepath}/train_contextual_preprocessed.csv", index=False)
        self.test.to_csv(f"{savepath}/test_contextual_preprocessed.csv", index=False)

        logger.info("Preprocessing for contextual embeddings completed.")