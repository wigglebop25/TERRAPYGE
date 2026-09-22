"""Baseline models for TERRAPYGE comparison.

Traditional ML models (Logistic Regression, Random Forest, XGBoost)
that use node features only (no graph structure) for landslide
susceptibility prediction. Used to demonstrate GNN superiority.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


class BaselineModel:
    """Base class for traditional ML baseline models.

    Args:
        name: Model name for display.
        config: Model-specific configuration dict.
    """

    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.model = None

    def train(self, X_train, y_train):
        """Train the model.

        Args:
            X_train: Training features [N_train, F].
            y_train: Training labels [N_train].
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """Predict class probabilities.

        Args:
            X: Features [N, F].

        Returns:
            probs: Probability of positive class [N].
        """
        raise NotImplementedError

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set.

        Args:
            X_test: Test features [N_test, F].
            y_test: Test labels [N_test].

        Returns:
            Dict with auc, ap, f1 metrics.
        """
        probs = self.predict_proba(X_test)
        preds = (probs > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_test, probs)
            ap = average_precision_score(y_test, probs)
        except ValueError:
            auc = 0.0
            ap = 0.0

        f1 = f1_score(y_test, preds, zero_division=0)

        return {
            'model': self.name,
            'auc': float(auc),
            'ap': float(ap),
            'f1': float(f1),
        }

    def save(self, path):
        """Save model to pickle file.

        Args:
            path: Path to save .pkl file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f'  {self.name} saved to {path}')

    def load(self, path):
        """Load model from pickle file.

        Args:
            path: Path to .pkl file.
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class LogisticRegressionBaseline(BaselineModel):
    """Logistic Regression baseline.

    Default params: max_iter=1000, random_state=42.
    """

    def __init__(self, config=None):
        super().__init__('Logistic Regression', config)
        cfg = config or {}
        self.model = LogisticRegression(
            max_iter=cfg.get('max_iter', 1000),
            random_state=42,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class RandomForestBaseline(BaselineModel):
    """Random Forest baseline.

    Default params: n_estimators=100, random_state=42.
    """

    def __init__(self, config=None):
        super().__init__('Random Forest', config)
        cfg = config or {}
        self.model = RandomForestClassifier(
            n_estimators=cfg.get('n_estimators', 100),
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBoostBaseline(BaselineModel):
    """XGBoost baseline.

    Default params: n_estimators=100, learning_rate=0.1, random_state=42.
    """

    def __init__(self, config=None):
        super().__init__('XGBoost', config)
        cfg = config or {}
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=cfg.get('n_estimators', 100),
                learning_rate=cfg.get('learning_rate', 0.1),
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
            )
        except ImportError:
            raise ImportError('XGBoost not installed. Run: pip install xgboost')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def run_baselines(X_train, y_train, X_test, y_test, config=None):
    """Run all baseline models and return comparison results.

    Args:
        X_train: Training features [N_train, F].
        y_train: Training labels [N_train].
        X_test: Test features [N_test, F].
        y_test: Test labels [N_test].
        config: Configuration dict with baseline params.

    Returns:
        results: Dict with metrics for each baseline model.
    """
    cfg = config or {}
    baseline_cfg = cfg.get('baselines', {})

    models = [
        LogisticRegressionBaseline(baseline_cfg.get('logistic_regression', {})),
        RandomForestBaseline(baseline_cfg.get('random_forest', {})),
        XGBoostBaseline(baseline_cfg.get('xgboost', {})),
    ]

    results = {}
    for model in models:
        print(f'\nTraining {model.name}...')
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        results[model.name] = metrics
        print(f'  AUC: {metrics["auc"]:.4f} | AP: {metrics["ap"]:.4f} | F1: {metrics["f1"]:.4f}')

    return results


def save_comparison(results, output_path):
    """Save baseline comparison results to JSON.

    Args:
        results: Dict with metrics for each model.
        output_path: Path to save JSON file.
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nComparison saved to {output_path}')
