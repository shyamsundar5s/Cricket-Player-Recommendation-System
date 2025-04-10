import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def train_model():
    # Sample dataset (replace with real cricket data)
    df = pd.read_csv('data/player_data.csv')

    X = df.drop(['Player', 'Rating'], axis=1)  # Features
    y = df['Rating']  # Target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    # Save model and scaler
    joblib.dump(model, 'mlp_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

def predict_player(features):
    model = joblib.load('mlp_model.pkl')
    scaler = joblib.load('scaler.pkl')
    scaled_features = scaler.transform([features])
    return model.predict(scaled_features)[0]
