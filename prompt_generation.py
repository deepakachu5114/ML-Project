import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import shap
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass
from datetime import datetime
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
from embeddings import TextEmbeddings


@dataclass
class ModelAnalysis:
    model_name: str
    performance_metrics: dict
    shap_patterns: dict
    error_analysis: dict
    prediction_distribution: dict


class ModelTracker:
    """Track model training metrics"""

    def __init__(self):
        self.losses = []

    def callback(self, env):
        self.losses.append(float(env.evaluation_result_list[0][1]))


class StressDetectionXAIOptimizer:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names

    def train_and_track_model(self, model, X_train, y_train, X_val, y_val) -> Tuple[Any, List[float]]:
        """Train model and track loss curve"""
        losses = []

        if isinstance(model, xgb.XGBClassifier):
            # For XGBoost, track loss at regular intervals
            n_estimators = model.get_params()['n_estimators']
            step = max(1, n_estimators // 10)  # Track loss every 10% of trees

            for i in range(step, n_estimators + 1, step):
                # Create and train a partial model
                model_partial = xgb.XGBClassifier(
                    **{**model.get_params(),
                       'n_estimators': i,
                       'random_state': 42,
                       'use_label_encoder': False}
                )
                model_partial.fit(X_train, y_train, verbose=False)

                # Calculate loss
                y_prob = model_partial.predict_proba(X_val)
                loss = log_loss(y_val, y_prob)
                losses.append(float(loss))

            # Finally, fit the full model
            model.fit(X_train, y_train, verbose=False)

        elif isinstance(model, RandomForestClassifier):
            # For Random Forest, manually calculate loss at intervals
            n_estimators = model.get_params()['n_estimators']
            step = max(1, n_estimators // 10)  # Track loss every 10% of trees

            # First fit the full model
            model.fit(X_train, y_train)

            # Then track partial fits for loss curve
            for i in range(step, n_estimators + 1, step):
                model_partial = RandomForestClassifier(
                    **{**model.get_params(), 'n_estimators': i, 'random_state': 42}
                )
                model_partial.fit(X_train, y_train)
                y_prob = model_partial.predict_proba(X_val)
                loss = log_loss(y_val, y_prob)
                losses.append(float(loss))

        return model, losses

    def analyze_shap_patterns(self, shap_values) -> Dict:
        """Feature importance and SHAP distribution"""
        global_importance = np.abs(shap_values.values).mean(0)
        feature_importance = dict(zip(self.feature_names, global_importance.tolist()))

        return {
            "feature_importance_ranking": feature_importance,
            "shap_value_distribution": {
                "mean": float(shap_values.values.mean()),
                "std": float(shap_values.values.std()),
                "max_impact": float(np.abs(shap_values.values).max()),
            },
        }

    def class_specific_feature_analysis(self, shap_values, y_pred) -> Dict:
        """Class-specific feature importance"""
        stress_shap = shap_values.values[y_pred == 1, :].mean(axis=0)
        no_stress_shap = shap_values.values[y_pred == 0, :].mean(axis=0)

        return {
            "stress": dict(zip(self.feature_names, stress_shap.tolist())),
            "no_stress": dict(zip(self.feature_names, no_stress_shap.tolist())),
        }

    def analyze_shap_variability(self, shap_values) -> Dict:
        """SHAP variability analysis"""
        variability = shap_values.values.std(axis=0)
        variability_dict = dict(zip(self.feature_names, variability.tolist()))
        return {"shap_variability": variability_dict}

    def analyze_errors(self, y_pred, probas) -> Dict:
        """Error analysis"""
        false_positives = ((y_pred == 1) & (self.y_test == 0)).sum()
        false_negatives = ((y_pred == 0) & (self.y_test == 1)).sum()
        confidence_scores = np.max(probas, axis=1)

        return {
            "error_counts": {
                "false_positives": int(false_positives),
                "false_negatives": int(false_negatives),
            },
            "prediction_confidence": {
                "mean_confidence": float(confidence_scores.mean()),
                "error_cases_confidence": float(
                    confidence_scores[(y_pred != self.y_test)].mean()
                ),
            },
        }

    def analyze_model(self, model, model_name: str, X_train, y_train) -> Dict:
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Train model and get loss curve
        trained_model, loss_curve = self.train_and_track_model(
            model, X_train_split, y_train_split, X_val, y_val
        )

        # Get predictions on test set
        y_pred = trained_model.predict(self.X_test)
        probas = trained_model.predict_proba(self.X_test)

        performance_metrics = {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred)),
            "recall": float(recall_score(self.y_test, y_pred)),
            "f1": float(f1_score(self.y_test, y_pred)),
            "loss_curve": loss_curve  # Add loss curve to metrics
        }

        explainer = shap.TreeExplainer(model, self.X_train)
        shap_values = explainer(self.X_test)

        analysis = {
            "model_name": model_name,
            "performance_metrics": performance_metrics,
            "shap_patterns": self.analyze_shap_patterns(shap_values),
            #"feature_interactions": self.analyze_feature_interactions(explainer, self.X_test),
            "class_specific_feature_analysis": self.class_specific_feature_analysis(
                shap_values, y_pred
            ),
            "shap_variability": self.analyze_shap_variability(shap_values),
            "error_analysis": self.analyze_errors(y_pred, probas),
            "prediction_distribution": {
                "stress_predicted": int(np.sum(y_pred == 1)),
                "no_stress_predicted": int(np.sum(y_pred == 0)),
            },
        }

        analysis["hyperparameter_insights"] = self.generate_hyperparameter_insights(model_name, model, analysis)

        return analysis

    def generate_hyperparameter_insights(self, model_name: str, model, analysis: dict) -> Dict:
        insights = {}
        if model_name == "XGBoost":
            insights = {
                "tree_structure": {
                    "max_depth": model.get_params()["max_depth"],
                    "min_child_weight": model.get_params()["min_child_weight"],
                    #"suggested_adjustments": []
                },
                "boosting_parameters": {
                    "learning_rate": model.get_params()["learning_rate"],
                    "n_estimators": model.get_params()["n_estimators"],
                    #"suggested_adjustments": []
                },
                "sampling_parameters": {
                    "colsample_bytree": model.get_params()["colsample_bytree"],
                    "subsample": model.get_params()["subsample"],
                    #"suggested_adjustments": []
                }
            }

        elif model_name == "RandomForest":
            insights = {
                "tree_structure": {
                    "max_depth": "None (unlimited)",
                    "min_samples_split": model.get_params()["min_samples_split"],
                    "min_samples_leaf": model.get_params()["min_samples_leaf"],
                    #"suggested_adjustments": []
                },
                "ensemble_parameters": {
                    "n_estimators": model.get_params()["n_estimators"],
                    "max_features": model.get_params()["max_features"],
                    #"suggested_adjustments": []
                }
            }

        return insights

    def generate_llm_prompt(self, model_name: str, analysis: Dict) -> str:
        """Generate structured JSON prompt for LLM optimization."""
        prompt = {
            "task": "stress_detection_optimization",
            "model_type": model_name,
            "current_state": {
                "performance_metrics": analysis["performance_metrics"],
                "analysis": {
                    "feature_importance": analysis["shap_patterns"]["feature_importance_ranking"],
                    "shap_value_distribution": analysis["shap_patterns"]["shap_value_distribution"],
                    #"feature_interactions": analysis["feature_interactions"]["feature_interactions"],
                    "class_specific_feature_analysis": analysis["class_specific_feature_analysis"],
                    "shap_variability": analysis["shap_variability"]["shap_variability"],
                    "error_analysis": analysis["error_analysis"],
                    "prediction_distribution": analysis["prediction_distribution"],
                    "hyperparameter_insights": analysis["hyperparameter_insights"],
                },
            },
            "optimization_request": """Based on this analysis, please suggest:
            1. Which hyperparameters should be adjusted and why
            2. The recommended direction of adjustment (increase/decrease)
            3. The reasoning behind each suggestion based on the SHAP analysis, feature interactions, variability, and error patterns.

            Please format your response as specific, actionable recommendations with numerical ranges where possible.
            Prioritize improvements in stress detection accuracy while maintaining interpretability and minimizing false negatives.""",
            "Output Format": """A JSON file containing suggested values of hyperparameters for each model with justification
                                Example:
                                {
                                Model1: {
                                            {
                                            hyperparam1: value1,
                                            Reason: justification1
                                            }
                                            {
                                            hyperparam2: value2,
                                            Reason: justification2...
                                        }
                                }"""
        }

        return json.dumps(prompt, indent=2)


def main():
    # Load your data
    train_data = pd.read_csv("train_noncontextual_preprocessed.csv")
    test_data = pd.read_csv("test_noncontextual_preprocessed.csv")

    # Initialize embeddings
    embeddings = TextEmbeddings(train_data, test_data)

    # Generate Word2Vec embeddings
    word2vec_train, word2vec_test = embeddings.apply_word2vec()

    # Load contextual data
    train_data_contextual = pd.read_csv("train_contextual_preprocessed.csv")
    test_data_contextual = pd.read_csv("test_contextual_preprocessed.csv")

    # Initialize contextual embeddings
    embeddings_contextual = TextEmbeddings(train_data_contextual, test_data_contextual)

    # Generate SentenceTransformer embeddings
    sentence_transformer_results = embeddings_contextual.apply_sentence_transformer()

    # Create feature names dictionary
    feature_names_dict = {'Word2Vec': embeddings.word2vec_features}

    for model_name, embeddings_result in sentence_transformer_results.items():
        if embeddings_result is not None:
            feature_names_dict[f'SentenceTransformer-{model_name}'] = embeddings_result['feature_names']

    # For this example, let's use Word2Vec embeddings
    X_train = word2vec_train
    X_test = word2vec_test
    y_train = train_data['label']
    y_test = test_data['label']
    feature_names = feature_names_dict['Word2Vec']

    # Initialize optimizer
    optimizer = StressDetectionXAIOptimizer(X_train, X_test, y_train, y_test, feature_names)

    # Initialize models with specified hyperparameters
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=1,
        colsample_bytree=1.0,
        subsample=1.0
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )

    # Analyze models
    xgb_analysis = optimizer.analyze_model(xgb_model, "XGBoost", X_train, y_train)
    rf_analysis = optimizer.analyze_model(rf_model, "RandomForest", X_train, y_train)

    # Generate LLM prompts
    xgb_prompt = optimizer.generate_llm_prompt("XGBoost", xgb_analysis)
    rf_prompt = optimizer.generate_llm_prompt("RandomForest", rf_analysis)

    # Save prompts to file
    with open('model_analysis_prompts.json', 'w') as f:
        json.dump({
            "xgboost_analysis": json.loads(xgb_prompt),
            "random_forest_analysis": json.loads(rf_prompt)
        }, f, indent=2)

    print("Analysis complete. Results saved to 'model_analysis_prompts.json'")


if __name__ == "__main__":
    main()