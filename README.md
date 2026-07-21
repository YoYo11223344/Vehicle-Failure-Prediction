# 🚗 Vehicle Failure Prediction Using Machine Learning

A production-ready Machine Learning application that predicts potential vehicle failures using sensor readings, maintenance history, usage patterns, and operating conditions. Built with Python, Streamlit, and Scikit-learn.

## Features

- **Dashboard** — KPI cards, feature importance, risk distribution, correlation heatmap, and interactive Plotly charts
- **Prediction** — User-friendly form with real-time failure prediction, confidence score, risk meter, and maintenance recommendations
- **Analytics** — Dataset preview, statistical summary, interactive visualizations, CSV upload for batch predictions, and downloadable results
- **About** — Project overview, workflow, technologies, ML models, and future scope

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn / Plotly
- Joblib

## Project Structure

```
VehicleFailurePrediction/
├── app.py              # Main Streamlit application
├── train_model.py      # Data generation & model training pipeline
├── utils.py            # Utility functions for predictions & preprocessing
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
├── data/               # Generated dataset (CSV)
├── model/              # Trained model & scaler (PKL)
├── pages/              # Reserved for future multi-page expansion
└── assets/             # Reserved for static assets
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd VehicleFailurePrediction

# Install dependencies
pip install -r requirements.txt

# Train the model (generates dataset + model files)
python train_model.py

# Run the application
streamlit run app.py
```

## How It Works

1. **Data Generation** — Synthetic vehicle sensor data (5,000+ records) with realistic failure patterns
2. **Preprocessing** — Missing value handling, label encoding, feature engineering, and standard scaling
3. **Model Training** — Random Forest and Gradient Boosting classifiers are trained and compared
4. **Evaluation** — Best model selected automatically based on F1 Score
5. **Prediction** — Real-time predictions with confidence scores and maintenance recommendations

## Deployment on Streamlit Cloud

1. Push the repository to GitHub
2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Click **"New app"** and select your repository
4. Set **Main file path** to `app.py`
5. Click **Deploy**

The app will automatically train models on first run if model files are missing.

## License

MIT
