# End-to-End Machine Learning Pipeline

A production-ready ML pipeline with modular components for data processing, model training, evaluation, and comparison.

## Features

- **Modular Architecture**: Separate classes for data processing, model creation, training, and persistence
- **Multiple Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Ensemble Learning**: Voting classifier combining multiple models
- **Cross-Validation**: Robust model evaluation with k-fold CV
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Feature Importance**: Extract and visualize feature importance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from pipeline import DataProcessor, ModelTrainer, ModelConfig

# Load and preprocess data
processor = DataProcessor()
X, y = processor.preprocess(df, target_col='target')

# Train model
config = ModelConfig(model_name='random_forest')
trainer = ModelTrainer(config)
trainer.train(X, y)

# Evaluate
results = trainer.evaluate()
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Cross-Validation

```python
cv_results = trainer.cross_validate(X, y)
print(f"CV Accuracy: {cv_results['mean_accuracy']:.4f}")
```

### Hyperparameter Tuning

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}
best = trainer.hyperparameter_search(X, y, param_grid)
print(f"Best params: {best['best_params']}")
```

### Model Persistence

```python
from pipeline import ModelPersistence

# Save
ModelPersistence.save_model(model, 'models/model.pkl')

# Load
model = ModelPersistence.load_model('models/model.pkl')
```

## Quick Start

```bash
python pipeline.py
```

## Project Structure

```
ml-pipeline/
├── pipeline.py          # Main pipeline code
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## Components

| Component | Description |
|-----------|-------------|
| `DataProcessor` | Data loading, cleaning, preprocessing |
| `ModelFactory` | Model creation and configuration |
| `ModelTrainer` | Training, CV, evaluation |
| `ModelPersistence` | Save/load models and results |

## Requirements

- Python 3.8+
- scikit-learn 1.3+
- NumPy, Pandas

## Author

Konstantinos Kantoutsis - [GitHub](https://github.com/kkanteea)

## License

MIT License
