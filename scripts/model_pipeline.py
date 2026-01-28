# Essential Libraries for Our ML Pipeline
import pandas as pd                    
import numpy as np                    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler   
from sklearn.ensemble import RandomForestClassifier 
from sklearn.pipeline import Pipeline          
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt   
import seaborn as sns               


def load_and_explore_data() -> tuple[np.array, np.ndarray, list, list]:
    """
    Load the Iris dataset and prepare it for ML pipeline.
    
    Returns:        
        tuple: Features (X), target labels (y), feature names, target names
    """
    iris = load_iris()
    
    # Create DataFrame for better data exploration
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({0: iris.target_names[0], 
                                     1: iris.target_names[1], 
                                     2: iris.target_names[2]})
    return iris.data, iris.target, iris.feature_names, iris.target_names

def create_train_test_split(X: np.array, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split dataset into training and testing sets with stratification.
    
    Args:
        X: Feature matrix
        y: Target labels  
        test_size: Proportion for testing (default: 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline():
    """
    Create preprocessing pipeline with a StandardScaler.
    """
    # TODO: Initialize and return a StandardScaler
    preprocessor = StandardScaler()
    return preprocessor

def create_model_pipeline(preprocessor):
    """
    Create complete ML pipeline: preprocessing → model training.
    
    Pipeline Steps:
    1. 'scaler': Standardize features using StandardScaler
    2. 'rf': Train RandomForest classifier
    
    Args:
        preprocessor: Preprocessing component (StandardScaler)
        
    Returns:
        sklearn.Pipeline: Complete ML pipeline
    """
    pipeline = Pipeline(
        [("scaler", preprocessor),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1))]
    )
    return pipeline

def train_model(pipeline, X_train, y_train):
    """
    Train the complete pipeline on training data.
    
    This automatically:
    1. Fits StandardScaler on training data
    2. Transforms training data  
    3. Trains RandomForest on scaled data
    """
    pipeline.fit(X_train, y_train)
    
    return pipeline 


def evaluate_model(pipeline, X_test, y_test):
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        pipeline: Trained ML pipeline
        X_test: Test features
        y_test: True test labels
        
    Returns:
        tuple: confusion_matrix, accuracy, precision, recall, f1_score
    """
    # Make predictions on test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate comprehensive metrics
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred, average="weighted")
    recall = recall_score(y_true=y_test, y_pred=y_pred, average="weighted")
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average="weighted")
    
    return cm, accuracy, precision, recall, f1

def visualize_results(cm, target_names, feature_importance_df, fig_path='./scripts/model_results.png'):
    """
    Create professional visualizations for model interpretation.
    
    Visualizations:
    1. Confusion Matrix: Shows prediction accuracy for each class
    2. Feature Importance: Reveals which features the model relies on most
    
    Args:
        cm: Confusion matrix
        target_names: Class names
        feature_importance_df: DataFrame with feature importance scores
        fig_path: Path to save the visualization
    """    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix\n(How often each class was predicted correctly)')
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('True Class')
    
    # 2. Feature Importance Bar Plot
    sns.barplot(data=feature_importance_df, x='importance', y='feature', ax=axes[1])
    axes[1].set_title('Feature Importance\n(Which features matter most for predictions)')
    axes[1].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def main():
    X, y, feature_names, target_names = load_and_explore_data()

    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    preprocessor = create_preprocessing_pipeline()
    pipeline = create_model_pipeline(preprocessor)
    pipeline = train_model(pipeline=pipeline, X_train=X_train, y_train=y_train)
    cm, accuracy, precision, recall, f1 = evaluate_model(pipeline=pipeline, X_test=X_test, y_test=y_test)

    feature_importance = pipeline.named_steps['rf'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    visualize_results(cm, target_names, feature_importance_df, fig_path='./scripts/model_results.png')

    return pipeline

if __name__ == "__main__":
    trained_model = main()