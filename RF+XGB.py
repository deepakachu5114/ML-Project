from preprocess import PreprocessForNonContextualEmbeddings, PreprocessForContextualEmbeddings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

'''# Preprocessing for non-contextual embeddings
preproc_noncontextual = PreprocessForNonContextualEmbeddings()
preproc_noncontextual.save()

# Preprocessing for contextual embeddings
preproc_contextual = PreprocessForContextualEmbeddings()
preproc_contextual.save()'''

# Generating embeddings
from embeddings import TextEmbeddings

# Load preprocessed data for non-contextual embeddings
train_data = pd.read_csv("train_noncontextual_preprocessed.csv")
test_data = pd.read_csv("test_noncontextual_preprocessed.csv")

# Initialize embeddings class for non-contextual data
embeddings = TextEmbeddings(train_data, test_data)

# Generate TF-IDF embeddings
tfidf_train, tfidf_test = embeddings.apply_tfidf()

# Generate Word2Vec embeddings
word2vec_train, word2vec_test = embeddings.apply_word2vec()

# Load preprocessed data for contextual embeddings
train_data_contextual = pd.read_csv("train_contextual_preprocessed.csv")
test_data_contextual = pd.read_csv("test_contextual_preprocessed.csv")

# Initialize embeddings class for contextual data
embeddings_contextual = TextEmbeddings(train_data_contextual, test_data_contextual)

# Generate SentenceTransformer embeddings
#sentence_transformer_train, sentence_transformer_test = embeddings_contextual.apply_sentence_transformer()
sentence_transformer_results = embeddings_contextual.apply_sentence_transformer()

def train_and_evaluate_models(X_train, y_train, X_test, y_test, embedding_name):
    # Split the training data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define models
    models = {
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        # Train the model
        if model_name == 'XGBoost':
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        else:
            model.fit(X_train, y_train)

        # Make predictions
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        def get_metrics(y_true, y_pred):
            return {
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'F1 Score': f1_score(y_true, y_pred),
                'Accuracy': accuracy_score(y_true, y_pred)
            }

        val_metrics = get_metrics(y_val, y_val_pred)
        test_metrics = get_metrics(y_test, y_test_pred)

        # Print results
        print(f"\nResults for {embedding_name} embeddings - {model_name}:")
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        results[model_name] = (model, test_metrics)

    return results


# Assuming you have your data and embeddings ready
# Run for each embedding type
all_results = {}

all_results['TF-IDF'] = train_and_evaluate_models(
    tfidf_train, train_data['label'], tfidf_test, test_data['label'], "TF-IDF")

all_results['Word2Vec'] = train_and_evaluate_models(
    word2vec_train, train_data['label'], word2vec_test, test_data['label'], "Word2Vec")

# For SentenceTransformer, we'll iterate through each model's results
for model_name, embeddings in sentence_transformer_results.items():
    if embeddings is not None:
        all_results[f'SentenceTransformer-{model_name}'] = train_and_evaluate_models(
            embeddings['train_embeddings'], train_data_contextual['label'],
            embeddings['test_embeddings'], test_data_contextual['label'],
            f"SentenceTransformer-{model_name}")

# Compare the results
results_df = pd.DataFrame({
    f"{emb_name}": metrics
    for emb_name, models in all_results.items()
    for model_name, (_, metrics) in models.items()
}).T

print("\nComparison of all models:")
print(results_df)

# Find the best model based on F1 Score
best_model = results_df['F1 Score'].idxmax()
print(f"\nBest model based on F1 Score: {best_model}")
print(f"Best F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f}")

# If you want to print processing times for SentenceTransformer models
print("\nProcessing times for SentenceTransformer models:")
for model_name, embeddings in sentence_transformer_results.items():
    if embeddings is not None:
        print(f"{model_name}: {embeddings['processing_time']:.2f} seconds")
