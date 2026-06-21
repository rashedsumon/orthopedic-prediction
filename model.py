import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_orthopedic_data

def train_pipeline():
    """
    Loads data, preprocesses it, trains a Random Forest model, 
    and returns the model, scaler, and test accuracy.
    """
    # 1. Load Data
    df = load_orthopedic_data()
    
    # Features (X) and Target Label (y)
    X = df.drop(columns=['class'])
    y = df['class']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
    
    # 3. Scale Features (Crucial for medical biomechanical metrics)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    accuracy = model.score(X_test_scaled, y_test)
    
    return model, scaler, accuracy

def predict_patient(model, scaler, input_features):
    """
    Takes raw numerical feature inputs, scales them, and returns a prediction.
    input_features should be a list or array of the 6 biomechanical metrics.
    """
    # Reshape for a single sample prediction
    features_array = np.array(input_features).reshape(1, -1)
    scaled_features = scaler.transform(features_array)
    
    prediction = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    return prediction, probabilities