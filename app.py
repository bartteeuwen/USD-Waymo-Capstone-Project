import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Autonomous Risk Triage | Waymo Capstone",
    page_icon="🚗",
    layout="wide"
)

# Title & Description
st.title("🚗 Autonomous Vehicle Scenario Risk Triage")
st.markdown("""
This dashboard ranks autonomous driving scene graphs based on predicted collision/criticality risk scores 
derived from **XGBoost (Tuned)** and **Graph Neural Network (GNN)** ensemble models.
""")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/triaged_scenarios.csv")
    return df

try:
    df = load_data()
    
    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Risk Filter Controls")
    
    risk_threshold = st.sidebar.slider(
        "Minimum Risk Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )
    
    filtered_df = df[df['predicted_risk_probability'] >= risk_threshold]
    
    # --- KPI Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluated Scenes", f"{len(df):,}")
    col2.metric("Triaged High-Risk Scenes", f"{len(filtered_df):,}", delta=f"{len(filtered_df)/len(df)*100:.1f}%")
    col3.metric("Max Predicted Risk Score", f"{df['predicted_risk_probability'].max():.4f}")
    col4.metric("Avg Risk Score (Filtered)", f"{filtered_df['predicted_risk_probability'].mean():.4f}" if len(filtered_df)>0 else "N/A")
    
    st.divider()
    
    # --- Data Table ---
    st.subheader("📋 Triaged High-Risk Scenarios")
    
    default_cols = [c for c in ['predicted_risk_probability', 'predicted_critical_class', 'min_inter_agent_dist', 'avg_agent_velocity', 'velocity_std'] if c in df.columns]
    selected_cols = st.multiselect("Select Display Columns", options=list(df.columns), default=default_cols)
    
    st.dataframe(
        filtered_df[selected_cols].style.highlight_max(axis=0, subset=['predicted_risk_probability'], color='#f8d7da'),
        use_container_width=True,
        height=400
    )

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.info("Ensure `data/triaged_scenarios.csv` exists in your repository.")
