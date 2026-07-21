import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'features.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vehicle_data.csv')

LE_ENCODE = {
    'road_condition': {'good': 0, 'average': 1, 'poor': 2},
    'driver_behavior': {'smooth': 0, 'moderate': 1, 'aggressive': 2},
}
LE_DECODE = {
    'road_condition': {v: k for k, v in LE_ENCODE['road_condition'].items()},
    'driver_behavior': {v: k for k, v in LE_ENCODE['driver_behavior'].items()},
}


def retrain_models():
    from train_model import train_and_save_models
    train_and_save_models()


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        retrain_models()
        return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    try:
        return joblib.load(SCALER_PATH)
    except Exception:
        retrain_models()
        return joblib.load(SCALER_PATH)


@st.cache_resource
def load_features():
    try:
        return joblib.load(FEATURES_PATH)
    except Exception:
        retrain_models()
        return joblib.load(FEATURES_PATH)


def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


def preprocess_input(df):
    df = df.copy()
    for col, mapping in LE_ENCODE.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def engineer_features(df):
    df = df.copy()
    df['temp_stress'] = df['engine_temp'] * df['ambient_temperature'] / 100
    df['wear_index'] = df['brake_pad_wear'] * (df['mileage'] / 100000)
    df['maintenance_gap'] = df['vehicle_age'] / (df['maintenance_frequency'] + 1)
    df['battery_age_risk'] = df['battery_voltage'] * (1 - df['vehicle_age'] / 20)
    df['pressure_ratio'] = df['oil_pressure'] / (df['tire_pressure'] + 1)
    return df


def predict_single(model, scaler, features, input_dict):
    df = pd.DataFrame([input_dict])
    df = preprocess_input(df)
    df = engineer_features(df)
    df = df[features]

    scaled = scaler.transform(df)
    pred = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    failure_prob = float(proba[1])
    return pred, failure_prob


def predict_batch(model, scaler, features, df):
    df = df.copy()
    df = preprocess_input(df)
    df = engineer_features(df)
    df = df[features]

    scaled = scaler.transform(df)
    preds = model.predict(scaled)
    probas = model.predict_proba(scaled)[:, 1]
    return preds, probas


def get_recommendation(proba):
    if proba < 0.30:
        return (
            "Vehicle is in good condition. Continue with your regular "
            "maintenance schedule."
        )
    elif proba < 0.60:
        return (
            "Vehicle shows early warning signs. Schedule a maintenance "
            "check-up soon to prevent potential issues."
        )
    else:
        return (
            "High risk of failure detected! Immediate inspection and "
            "repairs are strongly recommended."
        )


def get_confidence(proba):
    return max(proba, 1 - proba) * 100


def risk_level(proba):
    if proba < 0.30:
        return "Low", "#00e676"
    elif proba < 0.60:
        return "Medium", "#ffc107"
    else:
        return "High", "#ff1744"
