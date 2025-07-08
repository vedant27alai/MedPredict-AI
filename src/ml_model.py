from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(data):
    """Train multiple ML models, select the best, and print accuracy and confusion matrix."""
    features = ['price_deviation', 'manufacturer_reliability', 'is_rural']
    X = data[features]
    
    # Mock labels (1 = counterfeit, 0 = genuine) based on price deviation
    y = data['label'] = (
        (abs(data['price_deviation']) > 1).astype(int) +
        (data['manufacturer_reliability'] < 0.3).astype(int) +
        (data['is_rural'] == 1).astype(int)
    )
    data['label'] = (data['label'] >= 2).astype(int)  # Counterfeit if ≥2 issues
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    best_model = None
    best_accuracy = 0
    best_model_name = ''
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    # Train Isolation Forest for anomaly detection
    iso_model = IsolationForest(contamination=0.1, random_state=42)
    iso_model.fit(X)
    
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    return {'best': best_model, 'iso': iso_model}

def predict_risk(model, data):
    """Predict risk scores using the best model and Isolation Forest."""
    features = ['price_deviation', 'manufacturer_reliability', 'is_rural']
    X = data[features]
    
    # Combine best model and Isolation Forest predictions
    best_probs = model['best'].predict_proba(X)[:, 1]  # Probability of counterfeit
    iso_scores = model['iso'].decision_function(X)
    
    # Normalize Isolation Forest scores, avoiding division by zero
    if iso_scores.max() == iso_scores.min():
        norm_iso_scores = np.zeros_like(iso_scores)  # Set to 0 if all scores are identical
    else:
        norm_iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    
    return 0.7 * best_probs + 0.3 * (1 - norm_iso_scores)