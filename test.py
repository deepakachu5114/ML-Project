from preprocess import PreprocessForNonContextualEmbeddings, PreprocessForContextualEmbeddings
import pandas as pd
from constants import SAVEPATH


preproc_noncontextual = PreprocessForNonContextualEmbeddings()
preproc_noncontextual.save()

preproc_contextual = PreprocessForContextualEmbeddings()
preproc_contextual.save()

# generating embeddings

from embeddings import TextEmbeddings

# preprocessed data
train_data = pd.read_csv(f"{SAVEPATH}/train_noncontextual_preprocessed.csv")
test_data = pd.read_csv(f"{SAVEPATH}/test_noncontextual_preprocessed.csv")

embeddings = TextEmbeddings(train_data, test_data)

# tfidf embeddings
tfidf_train, tfidf_test = embeddings.apply_tfidf()

# word2vec embeddings
word2vec_train, word2vec_test = embeddings.apply_word2vec()

# stransformers embeddings
train_data_contextual = pd.read_csv(f"{SAVEPATH}/train_contextual_preprocessed.csv")
test_data_contextual = pd.read_csv(f"{SAVEPATH}/test_contextual_preprocessed.csv")
embeddings_contextual = TextEmbeddings(train_data_contextual, test_data_contextual)
sentence_transformer_train, sentence_transformer_test = embeddings_contextual.apply_sentence_transformer()
