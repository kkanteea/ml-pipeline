"""
End-to-End Machine Learning Pipeline
A production-ready ML pipeline with preprocessing, training, and evaluation
Author: Konstantinos Kantoutsis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_name: str
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names: List[str] = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
        return df
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data analysis report."""
        report = {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_stats': df.describe().to_dict()
        }
        return report
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        drop_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full preprocessing pipeline."""
        
        # Drop specified columns
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Impute missing values
        X = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=self.feature_names
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded
    
    def get_feature_importance(
        self, 
        model: Any, 
        top_n: int = 10
    ) -> pd.DataFrame:
        """Extract feature importance from trained model."""
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)


class ModelFactory:
    """Factory for creating ML models."""
    
    @staticmethod
    def get_model(name: str, **kwargs) -> Any:
        """Get model by name."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 5),
                random_state=kwargs.get('random_state', 42)
            ),
            'logistic_regression': LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            ),
            'svm': SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True,
                random_state=kwargs.get('random_state', 42)
            )
        }
        
        if name not in models:
            raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
        
        return models[name]
    
    @staticmethod
    def get_ensemble(**kwargs) -> VotingClassifier:
        """Create an ensemble of models."""
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]
        return VotingClassifier(estimators=estimators, voting='soft')


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.results: Dict[str, Any] = {}
        
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        model: Optional[Any] = None
    ) -> Any:
        """Train the model."""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Get model if not provided
        if model is None:
            model = ModelFactory.get_model(
                self.config.model_name,
                random_state=self.config.random_state
            )
        
        # Train
        print(f"Training {self.config.model_name}...")
        model.fit(X_train, y_train)
        self.model = model
        
        # Store test data for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return model
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation."""
        
        model = ModelFactory.get_model(
            self.config.model_name,
            random_state=self.config.random_state
        )
        
        scores = cross_val_score(
            model, X, y, 
            cv=self.config.cv_folds,
            scoring='accuracy',
            n_jobs=self.config.n_jobs
        )
        
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'all_scores': scores.tolist()
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate trained model."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(self.X_test)
        y_prob = None
        
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(
                self.y_test, y_pred, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }
        
        # AUC if binary classification
        if len(np.unique(self.y_test)) == 2 and y_prob is not None:
            results['roc_auc'] = roc_auc_score(self.y_test, y_prob[:, 1])
        
        self.results = results
        return results
    
    def hyperparameter_search(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        param_grid: Dict[str, List]
    ) -> Dict[str, Any]:
        """Grid search for hyperparameter tuning."""
        
        model = ModelFactory.get_model(self.config.model_name)
        
        grid_search = GridSearchCV(
            model, 
            param_grid,
            cv=self.config.cv_folds,
            scoring='accuracy',
            n_jobs=self.config.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }


class ModelPersistence:
    """Handles model saving and loading."""
    
    @staticmethod
    def save_model(model: Any, filepath: str) -> None:
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
    
    @staticmethod
    def save_results(results: Dict, filepath: str) -> None:
        """Save results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic classification data."""
    np.random.seed(42)
    
    # Features
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples) * 2,
        'feature_3': np.random.randn(n_samples) + 1,
        'feature_4': np.random.exponential(2, n_samples),
        'feature_5': np.random.uniform(0, 10, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on features
    score = (
        df['feature_1'] * 0.5 + 
        df['feature_2'] * 0.3 + 
        np.random.randn(n_samples) * 0.2
    )
    df['target'] = pd.cut(score, bins=3, labels=['low', 'medium', 'high'])
    
    # Add some missing values
    mask = np.random.random(n_samples) < 0.05
    df.loc[mask, 'feature_3'] = np.nan
    
    return df


def main():
    """Main execution function."""
    print("=" * 60)
    print("End-to-End ML Pipeline")
    print("=" * 60)
    
    # Generate data
    print("\n[1] Generating sample data...")
    df = generate_sample_data(1000)
    print(f"    Shape: {df.shape}")
    
    # Preprocess
    print("\n[2] Preprocessing...")
    processor = DataProcessor()
    analysis = processor.analyze_data(df)
    print(f"    Missing values: {sum(analysis['missing_values'].values())}")
    
    X, y = processor.preprocess(df, target_col='target')
    print(f"    Features shape: {X.shape}")
    print(f"    Classes: {np.unique(y)}")
    
    # Train multiple models
    print("\n[3] Training models...")
    models_to_train = ['random_forest', 'gradient_boosting', 'logistic_regression']
    results = {}
    
    for model_name in models_to_train:
        config = ModelConfig(model_name=model_name)
        trainer = ModelTrainer(config)
        
        # Cross-validation
        cv_results = trainer.cross_validate(X, y)
        print(f"\n    {model_name}:")
        print(f"    CV Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        
        # Full training
        trainer.train(X, y)
        eval_results = trainer.evaluate()
        
        results[model_name] = {
            'cv_accuracy': cv_results['mean_accuracy'],
            'test_accuracy': eval_results['accuracy'],
            'f1_score': eval_results['f1_score']
        }
    
    # Compare results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    # Best model
    best_model_name = results_df['test_accuracy'].idxmax()
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    
    # Feature importance
    print("\n[4] Feature Importance (Random Forest):")
    config = ModelConfig(model_name='random_forest')
    trainer = ModelTrainer(config)
    model = trainer.train(X, y)
    importance_df = processor.get_feature_importance(model)
    print(importance_df.to_string(index=False))
    
    # Ensemble
    print("\n[5] Training Ensemble...")
    ensemble = ModelFactory.get_ensemble()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    ensemble.fit(X_train, y_train)
    ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"    Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
