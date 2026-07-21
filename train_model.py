import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

DATA_PATH = os.path.join('data', 'vehicle_data.csv')
MODEL_PATH = os.path.join('model', 'model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
FEATURES_PATH = os.path.join('model', 'features.pkl')


def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    data = {
        'engine_temp': np.random.normal(90, 15, n_samples).clip(60, 130),
        'oil_pressure': np.random.normal(45, 10, n_samples).clip(15, 75),
        'brake_pad_wear': np.random.uniform(0, 100, n_samples),
        'tire_pressure': np.random.normal(32, 4, n_samples).clip(18, 48),
        'battery_voltage': np.random.normal(12.5, 1.0, n_samples).clip(9, 16),
        'fuel_efficiency': np.random.normal(15, 3, n_samples).clip(4, 28),
        'mileage': np.random.uniform(0, 200000, n_samples),
        'maintenance_frequency': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.10, 0.15, 0.25, 0.25, 0.15, 0.10]),
        'road_condition': np.random.choice(['good', 'average', 'poor'], n_samples, p=[0.40, 0.35, 0.25]),
        'driver_behavior': np.random.choice(['smooth', 'moderate', 'aggressive'], n_samples, p=[0.30, 0.40, 0.30]),
        'vehicle_age': np.random.uniform(0, 20, n_samples),
        'ambient_temperature': np.random.normal(25, 10, n_samples).clip(-10, 55),
        'vibration_level': np.random.exponential(0.5, n_samples).clip(0, 5),
        'coolant_level': np.random.normal(50, 10, n_samples).clip(10, 90),
        'transmission_slip': np.random.uniform(0, 10, n_samples),
    }
    df = pd.DataFrame(data)

    failure_prob = (
        0.15 * (df['engine_temp'] > 105).astype(int)
        + 0.12 * (df['oil_pressure'] < 30).astype(int)
        + 0.10 * (df['brake_pad_wear'] > 80).astype(int)
        + 0.08 * ((df['tire_pressure'] < 25) | (df['tire_pressure'] > 40)).astype(int)
        + 0.08 * (df['battery_voltage'] < 11.0).astype(int)
        + 0.07 * (df['fuel_efficiency'] < 10).astype(int)
        + 0.10 * (df['vibration_level'] > 1.5).astype(int)
        + 0.08 * (df['coolant_level'] < 30).astype(int)
        + 0.07 * (df['transmission_slip'] > 7).astype(int)
        + 0.05 * (df['maintenance_frequency'] < 1).astype(int)
        + 0.05 * (df['road_condition'] == 'poor').astype(int)
        + 0.05 * (df['driver_behavior'] == 'aggressive').astype(int)
        + 0.03 * (df['vehicle_age'] > 15).astype(int)
        + 0.02 * (df['mileage'] > 150000).astype(int)
    )

    noise = np.random.uniform(-0.10, 0.10, n_samples)
    failure_prob = np.clip(failure_prob + noise, 0, 1)
    df['failure'] = (failure_prob > 0.30).astype(int)

    return df


def preprocess_data(df):
    df = df.drop_duplicates().reset_index(drop=True)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0]) if not df[col].mode().empty else 'unknown'

    le = LabelEncoder()
    for col in ['road_condition', 'driver_behavior']:
        df[col] = le.fit_transform(df[col])

    return df


def engineer_features(df):
    df['temp_stress'] = df['engine_temp'] * df['ambient_temperature'] / 100
    df['wear_index'] = df['brake_pad_wear'] * (df['mileage'] / 100000)
    df['maintenance_gap'] = df['vehicle_age'] / (df['maintenance_frequency'] + 1)
    df['battery_age_risk'] = df['battery_voltage'] * (1 - df['vehicle_age'] / 20)
    df['pressure_ratio'] = df['oil_pressure'] / (df['tire_pressure'] + 1)
    return df


def train_and_save_models():
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    print("Generating synthetic dataset...")
    df = generate_synthetic_data(5000)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH} ({len(df)} records)")

    print("\nPreprocessing data...")
    df = preprocess_data(df)
    df = engineer_features(df)

    target = 'failure'
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(df)}")
    print(f"Failure rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    best_model = None
    best_f1 = 0
    best_name = ""

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba),
        }

        f1 = results[name]['F1 Score']
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    header = f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'ROC AUC':<10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        mark = "  <-- BEST" if name == best_name else ""
        print(
            f"{name:<22} {m['Accuracy']:<10.4f} {m['Precision']:<10.4f} "
            f"{m['Recall']:<10.4f} {m['F1 Score']:<10.4f} {m['ROC AUC']:<10.4f}{mark}"
        )
    print("=" * len(header))

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)

    print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {FEATURES_PATH}")

    return best_model, scaler, feature_cols


if __name__ == '__main__':
    train_and_save_models()
