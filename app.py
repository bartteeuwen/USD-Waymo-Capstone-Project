import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Public GCS Base URL for Rendered MP4s ---
GCS_VIDEO_BASE_URL = "https://storage.googleapis.com/waymo-capstone-rendered-videos"

# --- Page configuration ---
st.set_page_config(
    page_title="Autonomous Risk Triage | Waymo Capstone",
    page_icon="🚗",
    layout="wide"
)

# --- Title & Description ---
st.title("🚗 Autonomous Vehicle Scenario Risk Triage")
st.markdown("""
This dashboard ranks autonomous driving scene graphs based on predicted collision/criticality risk scores 
derived from **XGBoost (Tuned)** and **Graph Neural Network (GNN)** ensemble models.
""")

# --- Flexible Data Loader ---
@st.cache_data
def load_summary_data():
    # Order of paths to search
    possible_paths = [
        "data/high_risk_scenarios_valid.csv",
        "high_risk_scenarios_valid.csv",
        "data/triaged_scenarios.csv",
        "triaged_scenarios.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
            
    st.error("❌ Metadata CSV not found. Please place `high_risk_scenarios_valid.csv` or `triaged_scenarios.csv` in your app folder.")
    st.stop()

# --- Main App Logic ---
try:
    df = load_summary_data()
    
    # --- Data Cleaning & Normalization ---
    if 'scenario_id' in df.columns:
        df['scenario_id'] = df['scenario_id'].astype(str).str.strip().str.lower()
    
    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Risk Filter Controls")
    risk_threshold = st.sidebar.slider(
        "Minimum Risk Score Threshold",
        min_value=0.0, max_value=1.0, value=0.75, step=0.01
    )
    
    filtered_df = df.copy()
    if 'predicted_risk_probability' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['predicted_risk_probability'] >= risk_threshold]
        
    valid_physics_only = st.sidebar.checkbox("Exclude Physical Anomalies (> 38 m/s)", value=True)
    if valid_physics_only and 'is_valid_physics' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_valid_physics'] == True]
    
    # --- KPI Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluated Scenes", f"{len(df):,}")
    col2.metric("Triaged High-Risk Scenes", f"{len(filtered_df):,}", delta=f"{len(filtered_df)/len(df)*100:.1f}%" if len(df) > 0 else "0%")
    col3.metric("Max Predicted Risk Score", f"{df['predicted_risk_probability'].max():.4f}" if 'predicted_risk_probability' in df.columns else "N/A")
    col4.metric("Avg Risk Score (Filtered)", f"{filtered_df['predicted_risk_probability'].mean():.4f}" if len(filtered_df) > 0 and 'predicted_risk_probability' in filtered_df.columns else "N/A")
    
    st.divider()
    
    # --- Data Table ---
    st.subheader("📋 Triaged High-Risk Scenarios")
    default_cols = [c for c in ['scenario_id', 'predicted_risk_probability', 'target_risk_matrix', 'vehicle_count', 'pedestrian_count', 'max_velocity_mps', 'max_deceleration'] if c in df.columns]
    selected_cols = st.multiselect("Select Display Columns", options=list(df.columns), default=default_cols)
    
    if not filtered_df.empty and selected_cols:
        style_df = filtered_df[selected_cols]
        if 'predicted_risk_probability' in selected_cols:
            st.dataframe(
                style_df.style.highlight_max(axis=0, subset=['predicted_risk_probability'], color='#f8d7da'),
                use_container_width=True,
                height=280
            )
        else:
            st.dataframe(style_df, use_container_width=True, height=280)
    else:
        st.warning("No scenarios match your filter criteria or no display columns were selected.")

    st.divider()

    # --- Scenario Deep Dive Section ---
    st.subheader("🔍 Scenario Deep-Dive Inspector")
    
    if not filtered_df.empty:
        scenario_options = filtered_df['scenario_id'].unique()
        selected_scenario_id = st.selectbox(
            "Select Scenario ID to inspect:", 
            options=scenario_options
        )
        
        scene_info = df[df['scenario_id'] == selected_scenario_id].iloc[0]
        
        tab1, tab2 = st.tabs(["📊 Scene Breakdown", "🎥 Bird's Eye View (BEV) Playback"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**Agent Composition**")
                agent_counts = pd.DataFrame({
                    'Agent Type': ['Vehicles', 'Pedestrians', 'Cyclists'],
                    'Count': [
                        scene_info.get('vehicle_count', 0), 
                        scene_info.get('pedestrian_count', 0), 
                        scene_info.get('cyclist_count', 0)
                    ]
                })
                fig_agents = px.bar(agent_counts, x='Agent Type', y='Count', title="Agent Counts", color='Agent Type')
                st.plotly_chart(fig_agents, use_container_width=True)
                
            with col_b:
                st.write("**Map Infrastructure Elements**")
                map_counts = pd.DataFrame({
                    'Feature': ['Lanes', 'Stop Signs', 'Crosswalks', 'Speed Bumps'],
                    'Count': [
                        scene_info.get('lane_count', 0), 
                        scene_info.get('stop_sign_count', 0), 
                        scene_info.get('crosswalk_count', 0), 
                        scene_info.get('speed_bump_count', 0)
                    ]
                })
                fig_map = px.bar(map_counts, x='Feature', y='Count', title="Map Features")
                st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            st.markdown(f"**Streaming Pre-Rendered Trajectory Feed for Scenario:** `{selected_scenario_id}`")
            
            # Construct Public GCS Video URL
            video_url = f"{GCS_VIDEO_BASE_URL}/{selected_scenario_id}.mp4"
            
            col_vid, col_meta = st.columns([2, 1])
            
            with col_vid:
                st.video(video_url)
                st.caption(f"⚡ Streaming BEV video feed from GCS bucket for `{selected_scenario_id}`")
            
            with col_meta:
                st.markdown("### Scenario Highlights")
                st.metric("Risk Score", f"{scene_info.get('predicted_risk_probability', 0):.2%}")
                st.metric("Max Velocity", f"{scene_info.get('max_velocity_mps', 0):.1f} m/s")
                st.metric("Max Deceleration", f"{scene_info.get('max_deceleration', 0):.1f} m/s²")
                
    else:
        st.warning("No scenarios match the current filter criteria.")

except Exception as e:
    st.error(f"Error executing Streamlit dashboard: {e}")
