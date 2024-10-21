import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class TextEmbeddings:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def apply_tfidf(self):
        """
        Generate TF-IDF embeddings for the text data.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=1000)
        X_train = vectorizer.fit_transform(self.train_data['text'])
        X_test = vectorizer.transform(self.test_data['text'])

        return X_train.toarray(), X_test.toarray()

    def apply_word2vec(self, vector_size=100):
        """
        Generate Word2Vec embeddings for the text data.
        """
        from gensim.models import Word2Vec

        # Tokenize texts
        train_tokens = [text.split() for text in self.train_data['text']]
        test_tokens = [text.split() for text in self.test_data['text']]

        # Train Word2Vec model
        model = Word2Vec(sentences=train_tokens, vector_size=vector_size, window=5, min_count=1)

        # Generate document embeddings by averaging word vectors
        def get_doc_vector(tokens):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

        X_train = np.array([get_doc_vector(tokens) for tokens in train_tokens])
        X_test = np.array([get_doc_vector(tokens) for tokens in test_tokens])

        return X_train, X_test

    def apply_sentence_transformer(self, model_names=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
                                   train_batch_size=16, num_epochs=5):
        """
        Fine-tune transformer models and generate embeddings.
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        import evaluate
        import numpy as np

        def compute_metrics(eval_pred):
            accuracy_metric = evaluate.load("accuracy")
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return accuracy_metric.compute(predictions=predictions, references=labels)

        results = {}

        for model_name in model_names:
            try:
                start_time = time.time()

                # Initialize tokenizer and model
                model_path = f'sentence-transformers/{model_name}'
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=len(set(self.train_data['label'])),
                    problem_type="single_label_classification"
                )

                # Prepare datasets
                train_dataset = TextDataset(
                    self.train_data['text'].tolist(),
                    self.train_data['label'].tolist(),
                    tokenizer
                )

                eval_dataset = TextDataset(
                    self.test_data['text'].tolist(),
                    self.test_data['label'].tolist(),
                    tokenizer
                )

                # Define training arguments
                training_args = TrainingArguments(
                    output_dir=f'./results_{model_name}',
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=train_batch_size,
                    per_device_eval_batch_size=64,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir=f'./logs_{model_name}',
                    logging_steps=10,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    save_total_limit=1,
                    remove_unused_columns=False
                )

                # Initialize trainer with compute_metrics
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    compute_metrics=compute_metrics
                )

                # Fine-tune the model
                print(f"Fine-tuning {model_name}...")
                trainer.train()

                # Generate embeddings
                def get_embeddings(dataset):
                    embeddings = []
                    with torch.no_grad():
                        for i in range(len(dataset)):
                            inputs = {k: v.unsqueeze(0).to(model.device) for k, v in dataset[i].items()
                                      if k != 'labels'}
                            outputs = model(**inputs, output_hidden_states=True)
                            # Get the last hidden state
                            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                            embeddings.append(embedding[0])
                    return np.array(embeddings)

                # Generate embeddings for train and test sets
                X_train_embeddings = get_embeddings(train_dataset)
                X_test_embeddings = get_embeddings(eval_dataset)

                end_time = time.time()
                processing_time = end_time - start_time

                print(f"Model {model_name} fine-tuned and embeddings generated successfully.")
                print(f"Total processing time: {processing_time:.2f} seconds")

                results[model_name] = {
                    'train_embeddings': X_train_embeddings,
                    'test_embeddings': X_test_embeddings,
                    'processing_time': processing_time,
                    'model': model
                }
            except Exception as e:
                print(f"Error in fine-tuning/applying model {model_name}: {e}")
                results[model_name] = None

        return results


def train_and_evaluate_models(X_train, y_train, X_test, y_test, embedding_name):
    """
    Train and evaluate models using different embeddings.
    """
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


def main():
    # Load preprocessed data
    train_data = pd.read_csv("train_noncontextual_preprocessed.csv")
    test_data = pd.read_csv("test_noncontextual_preprocessed.csv")

    # Initialize embeddings class
    embeddings = TextEmbeddings(train_data, test_data)

    # Generate embeddings
    print("Generating TF-IDF embeddings...")
    tfidf_train, tfidf_test = embeddings.apply_tfidf()

    print("Generating Word2Vec embeddings...")
    word2vec_train, word2vec_test = embeddings.apply_word2vec()

    # Load contextual data
    train_data_contextual = pd.read_csv("train_contextual_preprocessed.csv")
    test_data_contextual = pd.read_csv("test_contextual_preprocessed.csv")

    # Initialize contextual embeddings
    embeddings_contextual = TextEmbeddings(train_data_contextual, test_data_contextual)

    # Generate and fine-tune transformer embeddings
    print("Fine-tuning transformer models and generating embeddings...")
    transformer_results = embeddings_contextual.apply_sentence_transformer(
        model_names=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
        train_batch_size=16,
        num_epochs=5
    )

    # Train and evaluate models
    all_results = {}

    all_results['TF-IDF'] = train_and_evaluate_models(
        tfidf_train, train_data['label'], tfidf_test, test_data['label'], "TF-IDF")

    all_results['Word2Vec'] = train_and_evaluate_models(
        word2vec_train, train_data['label'], word2vec_test, test_data['label'], "Word2Vec")

    # For transformer models, iterate through each model's results
    for model_name, embeddings in transformer_results.items():
        if embeddings is not None:
            all_results[f'Transformer-{model_name}'] = train_and_evaluate_models(
                embeddings['train_embeddings'], train_data_contextual['label'],
                embeddings['test_embeddings'], test_data_contextual['label'],
                f"Transformer-{model_name}")

    # Compare results
    results_df = pd.DataFrame({
        f"{emb_name}": metrics
        for emb_name, models in all_results.items()
        for model_name, (_, metrics) in models.items()
    }).T

    print("\nComparison of all models:")
    print(results_df)

    # Find best model
    best_model = results_df['F1 Score'].idxmax()
    print(f"\nBest model based on F1 Score: {best_model}")
    print(f"Best F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f}")

    # Print processing times for transformer models
    print("\nProcessing times for transformer models:")
    for model_name, embeddings in transformer_results.items():
        if embeddings is not None:
            print(f"{model_name}: {embeddings['processing_time']:.2f} seconds")


if __name__ == "__main__":
    main()