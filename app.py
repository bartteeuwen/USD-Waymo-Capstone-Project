import streamlit as st
import pandas as pd
import plotly.express as px

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

# Load Main Summary Dataset
@st.cache_data
def load_summary_data():
    return pd.read_csv("data/triaged_scenarios.csv")

# Helper function to load trajectory data only when requested
@st.cache_data
def load_scenario_trajectories(scenario_id):
    try:
        # Update "agent_trajectories.csv" to whatever your file is actually named!
        traj_df = pd.read_csv("data/agent_trajectories.csv") 
        
        # This filters the massive dataset down to just the single scenario you clicked on
        return traj_df[traj_df['scenario_id'] == scenario_id]
        
    except Exception as e:
        st.error(f"Could not load trajectory data: {e}")
        return None

try:
    df = load_summary_data()
    
    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Risk Filter Controls")
    
    risk_threshold = st.sidebar.slider(
        "Minimum Risk Score Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )
    
    # Filter anomalies toggle
    valid_physics_only = st.sidebar.checkbox("Exclude Physical Anomalies (> 38 m/s)", value=True)
    
    filtered_df = df[df['predicted_risk_probability'] >= risk_threshold]
    if valid_physics_only:
        filtered_df = filtered_df[filtered_df['is_valid_physics'] == True]
    
    # --- KPI Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluated Scenes", f"{len(df):,}")
    col2.metric("Triaged High-Risk Scenes", f"{len(filtered_df):,}", delta=f"{len(filtered_df)/len(df)*100:.1f}%")
    col3.metric("Max Predicted Risk Score", f"{df['predicted_risk_probability'].max():.4f}")
    col4.metric("Avg Risk Score (Filtered)", f"{filtered_df['predicted_risk_probability'].mean():.4f}" if len(filtered_df)>0 else "N/A")
    
    st.divider()
    
    # --- Data Table ---
    st.subheader("📋 Triaged High-Risk Scenarios")
    
    default_cols = [c for c in ['scenario_id', 'predicted_risk_probability', 'target_risk_matrix', 'min_inter_agent_dist', 'avg_agent_velocity', 'max_deceleration'] if c in df.columns]
    selected_cols = st.multiselect("Select Display Columns", options=list(df.columns), default=default_cols)
    
    st.dataframe(
        filtered_df[selected_cols].style.highlight_max(axis=0, subset=['predicted_risk_probability'], color='#f8d7da'),
        use_container_width=True,
        height=300
    )

    st.divider()

    # --- Scenario Deep Dive Section ---
    st.subheader("🔍 Scenario Deep-Dive Inspector")
    
    if not filtered_df.empty:
        # Let user select one scenario to inspect from the filtered list
        selected_scenario_id = st.selectbox(
            "Select Scenario ID to inspect:", 
            options=filtered_df['scenario_id'].unique()
        )
        
        scene_info = df[df['scenario_id'] == selected_scenario_id].iloc[0]
        
        tab1, tab2 = st.tabs(["📊 Scene Breakdown", "📍 Spatial Top-Down Replay"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**Agent Composition**")
                agent_counts = pd.DataFrame({
                    'Agent Type': ['Vehicles', 'Pedestrians', 'Cyclists'],
                    'Count': [scene_info['vehicle_count'], scene_info['pedestrian_count'], scene_info['cyclist_count']]
                })
                fig_agents = px.bar(agent_counts, x='Agent Type', y='Count', title="Agent Counts")
                st.plotly_chart(fig_agents, use_container_width=True)
                
            with col_b:
                st.write("**Map Infrastructure Elements**")
                map_counts = pd.DataFrame({
                    'Feature': ['Lanes', 'Stop Signs', 'Crosswalks', 'Speed Bumps'],
                    'Count': [scene_info['lane_count'], scene_info['stop_sign_count'], scene_info['crosswalk_count'], scene_info['speed_bump_count']]
                })
                fig_map = px.bar(map_counts, x='Feature', y='Count', title="Map Features")
                st.plotly_chart(fig_map, use_container_width=True)

        with tab2:
            st.markdown(f"**Trajectory Visualization for Scenario:** `{selected_scenario_id}`")
            
            # Load raw trajectory file when required
            traj_data = load_scenario_trajectories(selected_scenario_id)
            
            if traj_data is not None and not traj_data.empty:
                fig_replay = px.scatter(
                    traj_data, 
                    x='pos_x', 
                    y='pos_y', 
                    color='agent_type',
                    title=f"2D Agent Trajectories ({selected_scenario_id})"
                )
                st.plotly_chart(fig_replay, use_container_width=True)
            else:
                st.info("💡 Connect your trajectory dataframe in `load_scenario_trajectories()` to render the 2D spatial map here.")
    else:
        st.warning("No scenarios match the current filter criteria.")

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.info("Ensure `triaged_scenarios.csv` exists in your repository.")

import json
import streamlit as st
from google.cloud import storage

def fetch_raw_scenario(scenario_id, index_df):
    try:
        # 1. Look up the exact byte coordinates in the index
        target_info = index_df[index_df['scenario_id'] == scenario_id].iloc[0]
        
        # 2. Authenticate using Streamlit Secrets
        gcp_credentials = json.loads(st.secrets["GCP_KEY"])
        client = storage.Client.from_service_account_info(gcp_credentials)
        
        # Connect to the Waymo bucket
        bucket_name = "waymo_open_dataset_motion_v_1_2_0"
        bucket = client.bucket(bucket_name)
        
        # Clean the URI to get just the blob path
        blob_name = target_info['gcs_uri'].replace(f"gs://{bucket_name}/", "")
        blob = bucket.blob(blob_name)
        
        # 3. Calculate the byte range
        byte_start = target_info['byte_offset']
        byte_end = byte_start + target_info['byte_length'] - 1
        
        # 4. Fetch ONLY those specific bytes
        raw_bytes = blob.download_as_bytes(start=byte_start, end=byte_end)
        
        # 5. Strip the TFRecord header (8 bytes length + 4 bytes CRC)
        data_length = struct.unpack('<Q', raw_bytes[:8])[0]
        protobuf_bytes = raw_bytes[12 : 12 + data_length]
        
        # 6. Parse and return the full animated scenario!
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(protobuf_bytes)
        
        return scenario
        
    except Exception as e:
        st.error(f"Failed to fetch scenario: {e}")
        return None
