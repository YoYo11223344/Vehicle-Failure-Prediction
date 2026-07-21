import os
import sys
import warnings
import tempfile
from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils import (
    load_model, load_scaler, load_features, load_dataset,
    predict_single, predict_batch, get_recommendation, get_confidence, risk_level
)

st.set_page_config(
    page_title="Vehicle Failure Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

FORCE_DARK = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif !important; }

    body, .stApp, .stApp header, .main, .block-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        color: #e0e0e0 !important;
    }

    .block-container { padding: 2rem 1.5rem !important; }

    .glass {
        background: rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 20px !important;
        padding: 1.6rem 1.4rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35) !important;
    }

    .kpi-card {
        border-radius: 18px !important;
        padding: 1.5rem 1.2rem !important;
        color: #fff !important;
        text-align: center !important;
        box-shadow: 0 6px 24px rgba(0,0,0,0.30) !important;
    }
    .kpi-card .kpi-icon  { font-size: 2.2rem !important; margin-bottom: 0.3rem !important; }
    .kpi-card .kpi-value { font-size: 2.4rem !important; font-weight: 700 !important; line-height: 1.2 !important; color: #fff !important; }
    .kpi-card .kpi-label { font-size: 0.9rem !important; opacity: 0.85 !important; margin-top: 0.2rem !important; color: #fff !important; }

    .kpi-total   { background: linear-gradient(135deg, #667eea, #764ba2) !important; }
    .kpi-healthy { background: linear-gradient(135deg, #11998e, #38ef7d) !important; }
    .kpi-risk    { background: linear-gradient(135deg, #f093fb, #f5576c) !important; }
    .kpi-failure { background: linear-gradient(135deg, #fa709a, #fee140) !important; color: #222 !important; }
    .kpi-failure .kpi-value { color: #222 !important; }

    section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    .sidebar-title {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #fff !important;
        text-align: center !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.10) !important;
        margin-bottom: 1rem !important;
    }
    .sidebar-title span { background: linear-gradient(90deg, #667eea, #f093fb) !important; -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        color: #ccc !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.20) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #667eea22, #764ba222) !important;
        border-color: #667eea !important;
        color: #fff !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 8px 24px rgba(102,126,234,0.40) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05) !important;
        color: #aaa !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(102,126,234,0.20) !important;
        color: #fff !important;
        border-bottom: 2px solid #667eea !important;
    }

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.04) !important;
        font-weight: 600 !important;
    }

    .pred-healthy {
        background: linear-gradient(135deg, rgba(17,153,142,0.25), rgba(56,239,125,0.15)) !important;
        border: 1px solid rgba(56,239,125,0.30) !important;
    }
    .pred-failure {
        background: linear-gradient(135deg, rgba(240,147,251,0.20), rgba(245,87,108,0.20)) !important;
        border: 1px solid rgba(245,87,108,0.30) !important;
    }

    .risk-meter-container {
        background: rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
    }

    .footer {
        color: #e0e0e0 !important;
        border-top: 1px solid rgba(255,255,255,0.06) !important;
    }

    h1, h2, h3, h4, h5, h6 { color: #fff !important; }
    p, li, span, div, label { color: #e0e0e0 !important; }
    .stMarkdown, .stText { color: #e0e0e0 !important; }
</style>
<script>
    (function() {
        function forceDark() {
            document.body.setAttribute('data-theme', 'dark');
            document.documentElement.setAttribute('data-theme', 'dark');
            const vars = [
                ['--background-color', '#0f0c29'],
                ['--secondary-background-color', '#1a1a2e'],
                ['--text-color', '#e0e0e0'],
                ['--sidebar-background-color', '#0f0c29'],
            ];
            vars.forEach(([k, v]) => document.documentElement.style.setProperty(k, v));
        }
        forceDark();
        new MutationObserver(forceDark).observe(document.body, { attributes: true, attributeFilter: ['data-theme'] });
        new MutationObserver(forceDark).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    })();
</script>
"""


def apply_css():
    st.markdown(FORCE_DARK, unsafe_allow_html=True)


NAV_ITEMS = {
    "🏠  Dashboard": "Dashboard",
    "🔮  Prediction": "Prediction",
    "📊  Analytics": "Analytics",
    "ℹ️  About": "About",
}


def sidebar_nav():
    st.sidebar.markdown(
        '<div class="sidebar-title"><span>🚗 VFP</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    selected = st.sidebar.radio(
        "Navigate", list(NAV_ITEMS.keys()), label_visibility="collapsed"
    )

    st.sidebar.markdown("<br><hr style='border-color:rgba(255,255,255,0.06)'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='text-align:center;font-size:0.8rem;opacity:0.5;'>Vehicle Failure Prediction<br>ML‑powered diagnostics</div>",
        unsafe_allow_html=True,
    )
    return NAV_ITEMS[selected]


@st.cache_data
def load_data():
    df = load_dataset()
    if df is not None:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def render_kpi(label, value, icon, color_class, suffix=""):
    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}{suffix}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dashboard_page(df):
    st.markdown("<h2 style='color:#fff;'>📈 Dashboard</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-bottom:1.5rem;'>Real-time fleet overview and risk analytics</p>",
        unsafe_allow_html=True,
    )

    total = len(df)
    failures = int(df['failure'].sum())
    healthy = total - failures
    failure_rate = failures / total * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi("Total Vehicles", total, "🚗", "kpi-total")
    with col2:
        render_kpi("Healthy", healthy, "✅", "kpi-healthy")
    with col3:
        render_kpi("High Risk", failures, "⚠️", "kpi-risk")
    with col4:
        render_kpi("Failure Rate", f"{failure_rate:.1f}", "📊", "kpi-failure", "%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Feature Importance")
            model = load_model()
            features = load_features()
            base_feats = [c for c in features if c not in (
                'temp_stress', 'wear_index', 'maintenance_gap', 'battery_age_risk', 'pressure_ratio'
            )]
            if hasattr(model, 'feature_importances_'):
                fi = model.feature_importances_
                fi_df = pd.DataFrame({'Feature': features, 'Importance': fi})
                # group engineered features under originals for display
                fi_df = fi_df.sort_values('Importance', ascending=True).tail(12)
                fig = px.bar(
                    fi_df, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Viridis',
                    height=380,
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccc',
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 🥧 Risk Distribution")
            labels = ['Healthy', 'Failure Expected']
            values = [healthy, failures]
            colors = ['#38ef7d', '#f5576c']
            fig = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.55,
                marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0.3)', width=2)),
                textinfo='label+percent', textfont=dict(color='#fff', size=13),
            )])
            fig.update_layout(
                height=380,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccc',
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 🔥 Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.heatmap(
                corr, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.75},
            )
            ax.set_facecolor('none')
            fig.patch.set_visible(False)
            ax.tick_params(colors='#ccc', labelsize=8)
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color='#ccc')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ccc')
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 📉 Failure by Vehicle Age")
            age_bins = pd.cut(df['vehicle_age'], bins=10)
            age_fail = df.groupby(age_bins, observed=True)['failure'].mean().reset_index()
            age_fail.columns = ['Age Range', 'Failure Rate']
            age_fail['Age Range'] = age_fail['Age Range'].astype(str)
            fig = px.bar(
                age_fail, x='Age Range', y='Failure Rate',
                color='Failure Rate', color_continuous_scale='Reds',
                height=380,
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccc',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Vehicle Age",
                yaxis_title="Failure Rate",
            )
            fig.update_xaxes(showgrid=False, tickangle=45)
            fig.update_yaxes(showgrid=False, tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


def prediction_page(df):
    st.markdown("<h2 style='color:#fff;'>🔮 Vehicle Failure Prediction</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-bottom:1.5rem;'>Enter vehicle parameters below to predict potential failure risk.</p>",
        unsafe_allow_html=True,
    )

    model = load_model()
    scaler = load_scaler()
    features = load_features()

    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("##### ⚙️ Vehicle Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            engine_temp = st.slider("Engine Temp (°C)", 60, 130, 90, help="Normal range: 80–100°C")
            oil_pressure = st.slider("Oil Pressure (PSI)", 15, 75, 45, help="Normal range: 35–55 PSI")
            brake_pad_wear = st.slider("Brake Pad Wear (%)", 0, 100, 30, help="Replace above 80%")
            tire_pressure = st.slider("Tire Pressure (PSI)", 18, 48, 32, help="Normal: 30–35 PSI")
            battery_voltage = st.slider("Battery Voltage (V)", 9.0, 16.0, 12.5, 0.1, help="Normal: 12.4–12.8V")

        with col2:
            fuel_efficiency = st.slider("Fuel Efficiency (km/L)", 4.0, 28.0, 15.0, 0.5)
            mileage = st.slider("Mileage (km)", 0, 200000, 50000, 1000)
            maintenance_freq = st.slider("Maintenance (times/yr)", 0, 5, 2, help="How often vehicle is serviced")
            vehicle_age = st.slider("Vehicle Age (years)", 0.0, 20.0, 5.0, 0.5)
            ambient_temp = st.slider("Ambient Temp (°C)", -10, 55, 25)

        with col3:
            vibration_level = st.slider("Vibration Level (g)", 0.0, 5.0, 0.5, 0.1, help="Higher values indicate issues")
            coolant_level = st.slider("Coolant Level (%)", 10, 90, 50, help="Normal: 40–60%")
            transmission_slip = st.slider("Transmission Slip", 0.0, 10.0, 2.0, 0.5)
            road_condition = st.selectbox("Road Condition", ["good", "average", "poor"])
            driver_behavior = st.selectbox("Driver Behavior", ["smooth", "moderate", "aggressive"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Failure Risk")

        st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        input_data = {
            'engine_temp': engine_temp,
            'oil_pressure': oil_pressure,
            'brake_pad_wear': brake_pad_wear,
            'tire_pressure': tire_pressure,
            'battery_voltage': battery_voltage,
            'fuel_efficiency': fuel_efficiency,
            'mileage': mileage,
            'maintenance_frequency': maintenance_freq,
            'road_condition': road_condition,
            'driver_behavior': driver_behavior,
            'vehicle_age': vehicle_age,
            'ambient_temperature': ambient_temp,
            'vibration_level': vibration_level,
            'coolant_level': coolant_level,
            'transmission_slip': transmission_slip,
        }

        with st.spinner("Analyzing vehicle data..."):
            pred, proba = predict_single(model, scaler, features, input_data)

        confidence = get_confidence(proba)
        risk_text, risk_color = risk_level(proba)
        rec = get_recommendation(proba)

        status = "✅ Healthy" if pred == 0 else "❌ Failure Expected"
        status_class = "pred-healthy" if pred == 0 else "pred-failure"
        status_color = "#38ef7d" if pred == 0 else "#f5576c"

        col_r1, col_r2 = st.columns([1, 1])

        with col_r1:
            st.markdown(
                f"""
                <div class="pred-box {status_class}">
                    <div class="pred-label">Prediction Result</div>
                    <div class="pred-status" style="color:{status_color};">{status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r2:
            st.markdown(
                f"""
                <div class="pred-box {status_class}">
                    <div class="pred-label">Failure Probability</div>
                    <div class="pred-status" style="color:{status_color};">{proba:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"##### Risk Level: <span style='color:{risk_color};'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="risk-meter-container">
                <div class="risk-meter-fill" style="width:{proba*100:.0f}%;background:linear-gradient(90deg, #38ef7d, #ffc107, #f5576c);"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**Confidence Score:** {confidence:.1f}% &nbsp;|&nbsp; **Model:** {type(model).__name__}",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border-radius: 14px;
                padding: 1rem 1.5rem;
                margin-top: 1rem;
                border-left: 4px solid {risk_color};
            ">
                <strong>💡 Recommendation:</strong> {rec}
            </div>
            """,
            unsafe_allow_html=True,
        )


def analytics_page(df):
    st.markdown("<h2 style='color:#fff;'>📊 Analytics & Insights</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#aaa;margin-bottom:1.5rem;'>Explore the dataset, visualize trends, and run batch predictions.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["📋 Dataset Preview", "📈 Visualizations", "📤 Batch Prediction"])

    with tabs[0]:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown(f"##### Dataset Overview — {len(df)} Records")

            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Rows", f"{len(df):,}")
            with col_info2:
                st.metric("Columns", len(df.columns))
            with col_info3:
                st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
            with col_info4:
                st.metric("Categorical Features", len(df.select_dtypes(include=['object']).columns))

            st.dataframe(df.head(100), use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("##### 📊 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            with st.container():
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.markdown("##### Engine Temp vs Failure")
                fig = px.histogram(
                    df, x='engine_temp', color='failure',
                    barmode='overlay', opacity=0.65, nbins=40,
                    color_discrete_map={0: '#38ef7d', 1: '#f5576c'},
                    labels={'failure': 'Status', 'engine_temp': 'Engine Temperature (°C)'},
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccc', height=320, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with col_v2:
            with st.container():
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.markdown("##### Oil Pressure vs Failure")
                fig = px.histogram(
                    df, x='oil_pressure', color='failure',
                    barmode='overlay', opacity=0.65, nbins=40,
                    color_discrete_map={0: '#38ef7d', 1: '#f5576c'},
                    labels={'failure': 'Status', 'oil_pressure': 'Oil Pressure (PSI)'},
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccc', height=320, margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        col_v3, col_v4 = st.columns(2)

        with col_v3:
            with st.container():
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.markdown("##### Brake Pad Wear Distribution")
                fig = px.box(
                    df, x='failure', y='brake_pad_wear', color='failure',
                    color_discrete_map={0: '#38ef7d', 1: '#f5576c'},
                    labels={'failure': 'Status', 'brake_pad_wear': 'Brake Pad Wear (%)'},
                    points=False,
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccc', height=320, margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                fig.update_xaxes(showgrid=False, tickvals=[0, 1], ticktext=['Healthy', 'Failure'])
                fig.update_yaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with col_v4:
            with st.container():
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                st.markdown("##### Maintenance Frequency Impact")
                maint_fail = df.groupby('maintenance_frequency')['failure'].mean().reset_index()
                maint_fail.columns = ['Maintenance (per year)', 'Failure Rate']
                fig = px.line(
                    maint_fail, x='Maintenance (per year)', y='Failure Rate',
                    markers=True, line_shape='spline',
                    color_discrete_sequence=['#667eea'],
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccc', height=320, margin=dict(l=10, r=10, t=10, b=10),
                )
                fig.update_xaxes(showgrid=False, tickmode='linear', dtick=1)
                fig.update_yaxes(showgrid=False, tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("##### 📊 Pairwise Sensor Relationships")
            sensor_cols = ['engine_temp', 'oil_pressure', 'brake_pad_wear', 'tire_pressure', 'battery_voltage']
            fig = px.scatter_matrix(
                df, dimensions=sensor_cols, color='failure',
                color_discrete_map={0: '#38ef7d', 1: '#f5576c'},
                opacity=0.35, height=600,
            )
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccc', margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("##### 📤 Upload CSV for Batch Prediction")

            uploaded_file = st.file_uploader(
                "Choose a CSV file", type=['csv'],
                help="File must include all sensor/parameter columns matching the training data."
            )

            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.success(f"Uploaded: {uploaded_file.name} ({len(batch_df)} records)")

                    model = load_model()
                    scaler = load_scaler()
                    features = load_features()

                    required = [c for c in features if c not in (
                        'temp_stress', 'wear_index', 'maintenance_gap', 'battery_age_risk', 'pressure_ratio'
                    )]
                    missing = [c for c in required if c not in batch_df.columns]
                    if missing:
                        st.error(f"Missing columns: {', '.join(missing)}")
                    else:
                        with st.spinner("Running batch predictions..."):
                            preds, probas = predict_batch(model, scaler, features, batch_df)

                        batch_df['Prediction'] = ['Healthy' if p == 0 else 'Failure Expected' for p in preds]
                        batch_df['Failure Probability'] = probas.round(4)
                        batch_df['Risk Level'] = [
                            'Low' if p < 0.3 else 'Medium' if p < 0.6 else 'High' for p in probas
                        ]
                        batch_df['Recommendation'] = [get_recommendation(p) for p in probas]

                        st.dataframe(batch_df, use_container_width=True, height=400)

                        fail_count = int(sum(preds))
                        st.markdown(
                            f"**Results:** {len(preds) - fail_count} Healthy, "
                            f"{fail_count} Failure Expected "
                            f"({fail_count / len(preds) * 100:.1f}% failure rate)"
                        )

                        csv_buffer = StringIO()
                        batch_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv_buffer.getvalue(),
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            else:
                st.info("Upload a CSV file with vehicle parameters to run batch predictions. You can download the sample dataset from the Dataset Preview tab.")

            st.markdown("</div>", unsafe_allow_html=True)


def about_page():
    st.markdown("<h2 style='color:#fff;'>ℹ️ About</h2>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("""
        ### 🚗 Vehicle Failure Prediction System

        A production-ready Machine Learning application that predicts potential vehicle
        failures using sensor readings, maintenance history, usage patterns, and
        operating conditions. Built with **Python**, **Streamlit**, and **Scikit-learn**.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 🔄 Workflow")
            st.markdown("""
            1. **Data Generation** — Synthetic vehicle sensor data (5,000+ records)
            2. **Preprocessing** — Missing values, encoding, feature engineering, scaling
            3. **Model Training** — Multiple classifiers trained and compared
            4. **Evaluation** — Best model selected via F1 score
            5. **Deployment** — Streamlit frontend with real-time predictions
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_a2:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 🛠️ Technologies Used")
            st.markdown("""
            - **Python** — Core programming language
            - **Streamlit** — Interactive web UI framework
            - **Pandas / NumPy** — Data manipulation
            - **Scikit-learn** — ML models & preprocessing
            - **Plotly / Matplotlib / Seaborn** — Visualizations
            - **Joblib** — Model serialization
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 🤖 ML Models")
            st.markdown("""
            | Model | Description |
            |---|---|
            | **Random Forest** | Ensemble of decision trees — robust & accurate |
            | **Gradient Boosting** | Sequential ensemble for high accuracy |

            The best model is selected automatically based on **F1 Score**.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_b2:
        with st.container():
            st.markdown("<div class='glass'>", unsafe_allow_html=True)
            st.markdown("#### 🔮 Future Scope")
            st.markdown("""
            - **Real-time sensor integration** via IoT/API
            - **Anomaly detection** for early warning system
            - **Explainable AI** (SHAP / LIME) for predictions
            - **Fleet-wide analytics** dashboard
            - **Mobile app** with push notifications
            - **Historical trend analysis** with time-series models
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;'>"
            "Made with ❤️ using Streamlit &nbsp;|&nbsp; "
            "Vehicle Failure Prediction System &nbsp;|&nbsp; "
            "© 2025"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    apply_css()

    page = sidebar_nav()

    df = load_data()

    if df is None:
        st.error("Dataset not found. Run `python train_model.py` first to generate data and train the model.")
        st.info("The training script will generate a synthetic dataset and train all models.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training models... This may take a minute."):
                from train_model import train_and_save_models
                train_and_save_models()
            st.success("Model training complete! Please refresh the app.")
            st.rerun()
        return

    model_available = (
        os.path.exists(os.path.join(BASE_DIR, 'model', 'model.pkl'))
        and os.path.exists(os.path.join(BASE_DIR, 'model', 'scaler.pkl'))
    )

    if not model_available:
        st.error("Model files not found. Run `python train_model.py` to train and save the model.")
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training models..."):
                from train_model import train_and_save_models
                train_and_save_models()
            st.success("Model training complete! Please refresh the app.")
            st.rerun()
        return

    try:
        if page == "Dashboard":
            dashboard_page(df)
        elif page == "Prediction":
            prediction_page(df)
        elif page == "Analytics":
            analytics_page(df)
        elif page == "About":
            about_page()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

    st.markdown("<div class='footer'>🚗 Vehicle Failure Prediction System — Powered by Machine Learning</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
