import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import shap
import scipy.sparse
import traceback

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
#sentence_transformer_results = embeddings_contextual.apply_sentence_transformer()

# Create feature names dictionary
feature_names_dict = {}
feature_names_dict['TF-IDF'] = embeddings.tfidf_vectorizer.get_feature_names_out()
feature_names_dict['Word2Vec'] = embeddings.word2vec_features

# Generate SentenceTransformer embeddings and store feature names
sentence_transformer_results = embeddings_contextual.apply_sentence_transformer()
for model_name, embeddings_result in sentence_transformer_results.items():
    if embeddings_result is not None:
        feature_names_dict[f'SentenceTransformer-{model_name}'] = embeddings_result['feature_names']

def train_and_evaluate_models(X_train, y_train, X_test, y_test, embedding_name, feature_names_dict):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    models = {
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        # Train the model
        if model_name == 'XGBoost':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        def get_metrics(y_true, y_pred):
            return {
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'F1 Score': f1_score(y_true, y_pred),
                'Accuracy': accuracy_score(y_true, y_pred)
            }

        val_metrics = get_metrics(y_val, y_val_pred)
        test_metrics = get_metrics(y_test, y_test_pred)

        # Print metrics
        print(f"\nResults for {embedding_name} embeddings - {model_name}:")
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nTest Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Add SHAP analysis
        print(f"\nGenerating SHAP values for {embedding_name} - {model_name}...")
        try:
            ind = 6
            current_feature_names = feature_names_dict[embedding_name]
            predicted_class = model.predict(X_test[ind].reshape(1, -1))[0]


            # Create SHAP explainer
            explainer = shap.Explainer(model, X_train, feature_names=current_feature_names)

            shap_values = explainer(X_test)

            print(f"SHAP values shape: {shap_values.values.shape}")

            # Generate waterfall plot for a sample instance
            print(f"\nAnalyzing test instance {ind}:")
            print(f"Predicted class: {predicted_class}")
            if hasattr(test_data, 'iloc'):
                print(f"Original text: {test_data.iloc[ind]['text']}")

            shap.initjs()
            if len(shap_values.shape) == 3:
                shap.plots.waterfall(shap_values[ind, :, predicted_class])
            else:
                shap.plots.waterfall(shap_values[ind])

            # Store results
            results[model_name] = {
                'model': model,
                'metrics': test_metrics,
                'shap_values': shap_values
            }

        except Exception as e:
            print(f"Error generating SHAP values: {str(e)}")
            results[model_name] = {
                'model': model,
                'metrics': test_metrics,
                'shap_values': None
            }

    return results


# Assuming you have your data and embeddings ready
# Run for each embedding type
all_results = {}

all_results['TF-IDF'] = train_and_evaluate_models(
    tfidf_train, train_data['label'], tfidf_test, test_data['label'], "TF-IDF", feature_names_dict)

all_results['Word2Vec'] = train_and_evaluate_models(
    word2vec_train, train_data['label'], word2vec_test, test_data['label'], "Word2Vec", feature_names_dict)

# For SentenceTransformer, we'll iterate through each model's results
for model_name, embeddings in sentence_transformer_results.items():
    if embeddings is not None:
        all_results[f'SentenceTransformer-{model_name}'] = train_and_evaluate_models(
            embeddings['train_embeddings'], train_data_contextual['label'],
            embeddings['test_embeddings'], test_data_contextual['label'],
            f"SentenceTransformer-{model_name}", feature_names_dict)

# And modify how you process the results
results_df = pd.DataFrame({
    f"{emb_name}-{model_name}": metrics['metrics']
    for emb_name, models in all_results.items()
    for model_name, metrics in models.items()
}).T
results_df.to_csv("Comparisons.csv", index=False)

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
