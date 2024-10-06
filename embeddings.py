import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from logger_config import logger

class TextEmbeddings:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        logger.info("TextEmbeddings initialized.")

    def apply_tfidf(self):
        try:
            vectorizer = TfidfVectorizer()
            X_train_tfidf = vectorizer.fit_transform(self.train_data['text'])
            X_test_tfidf = vectorizer.transform(self.test_data['text'])

            # Convert to DataFrame for better visualization
            tfidf_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
            tfidf_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
            logger.info("TF-IDF embeddings generated successfully.")
            return tfidf_train_df, tfidf_test_df
        except Exception as e:
            logger.error(f"Error in applying TF-IDF: {e}")
            return None, None


    def apply_word2vec(self):
        # Tokenize the text for Word2Vec
        try:
            train_tokens = [text.split() for text in self.train_data['text']]
            test_tokens = [text.split() for text in self.test_data['text']]

            # Train Word2Vec model
            model = Word2Vec(sentences=train_tokens, vector_size=100, window=5, min_count=1, workers=4)

            # Get vector for each sentence (average of word vectors)
            X_train_word2vec = np.array([np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
                                         for tokens in train_tokens])
            X_test_word2vec = np.array([np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
                                        for tokens in test_tokens])
            logger.info("Word2Vec embeddings generated successfully.")
            return X_train_word2vec, X_test_word2vec
        except Exception as e:
            logger.error(f"Error in applying Word2Vec: {e}")
            return None, None


    def apply_sentence_transformer(self, model_name='all-MiniLM-L6-v2'):
        """
        Generate sentence embeddings using a SentenceTransformer model.
        """
        try:
            model = SentenceTransformer(model_name)
            X_train_embeddings = model.encode(self.train_data['text'].tolist(), convert_to_tensor=True)
            X_test_embeddings = model.encode(self.test_data['text'].tolist(), convert_to_tensor=True)
            logger.info("SentenceTransformer embeddings generated successfully.")
            return X_train_embeddings.cpu().numpy(), X_test_embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error in applying SentenceTransformer: {e}")
            return None, None
