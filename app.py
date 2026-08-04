import os
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

# --- Load Main Data & Video Index ---
@st.cache_data
def load_datasets():
    # 1. Load Main Triage Summary (Full Dataset)
    main_paths = ["data/triaged_scenarios.csv", "triaged_scenarios.csv", "data/high_risk_scenarios.csv", "high_risk_scenarios.csv"]
    df_main = None
    for path in main_paths:
        if os.path.exists(path):
            df_main = pd.read_csv(path)
            break
            
    # 2. Load Validated Video Scenarios (20-row subset)
    valid_paths = ["data/high_risk_scenarios_valid.csv", "high_risk_scenarios_valid.csv"]
    df_valid = None
    for path in valid_paths:
        if os.path.exists(path):
            df_valid = pd.read_csv(path)
            break
            
    # Fallback logic if only one file exists
    if df_main is None and df_valid is not None:
        df_main = df_valid
    elif df_main is None:
        st.error("❌ No scenario CSV dataset found in repository.")
        st.stop()
        
    return df_main, df_valid

# --- Main Execution ---
try:
    df_summary, df_videos = load_datasets()
    
    # Clean IDs
    if 'scenario_id' in df_summary.columns:
        df_summary['scenario_id'] = df_summary['scenario_id'].astype(str).str.strip().str.lower()
    if df_videos is not None and 'scenario_id' in df_videos.columns:
        df_videos['scenario_id'] = df_videos['scenario_id'].astype(str).str.strip().str.lower()

    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Risk Filter Controls")
    risk_threshold = st.sidebar.slider(
        "Minimum Risk Score Threshold",
        min_value=0.0, max_value=1.0, value=0.75, step=0.01
    )
    
    filtered_df = df_summary.copy()
    if 'predicted_risk_probability' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['predicted_risk_probability'] >= risk_threshold]
        
    valid_physics_only = st.sidebar.checkbox("Exclude Physical Anomalies (> 38 m/s)", value=True)
    if valid_physics_only and 'is_valid_physics' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_valid_physics'] == True]
    
    # --- KPI Summary Cards (Reflect Full Dataset) ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluated Scenes", f"{len(df_summary):,}")
    col2.metric("Triaged High-Risk Scenes", f"{len(filtered_df):,}", delta=f"{len(filtered_df)/len(df_summary)*100:.1f}%" if len(df_summary) > 0 else "0%")
    col3.metric("Max Predicted Risk Score", f"{df_summary['predicted_risk_probability'].max():.4f}" if 'predicted_risk_probability' in df_summary.columns else "N/A")
    col4.metric("Avg Risk Score (Filtered)", f"{filtered_df['predicted_risk_probability'].mean():.4f}" if len(filtered_df) > 0 and 'predicted_risk_probability' in filtered_df.columns else "N/A")
    
    st.divider()
    
    # --- Data Table (Shows Full Triaged List) ---
    st.subheader("📋 Triaged High-Risk Scenarios")
    default_cols = [c for c in ['scenario_id', 'predicted_risk_probability', 'target_risk_matrix', 'vehicle_count', 'pedestrian_count', 'max_velocity_mps', 'max_deceleration'] if c in df_summary.columns]
    selected_cols = st.multiselect("Select Display Columns", options=list(df_summary.columns), default=default_cols)
    
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

    # --- Scenario Deep Dive Section (Restricted to Rendered Subset) ---
    st.subheader("🔍 Scenario Deep-Dive Inspector")
    
    # Determine options with available videos
    if df_videos is not None:
        video_available_ids = set(df_videos['scenario_id'].unique())
        # Filter dropdown options to ONLY scenes that have rendered videos
        renderable_options = [sid for sid in filtered_df['scenario_id'].unique() if sid in video_available_ids]
        
        # Fallback to all video IDs if current slider filter excluded them
        if not renderable_options:
            renderable_options = list(video_available_ids)
            
        st.info("💡 **Deep-Dive Replay Available:** The selector below is filtered to scenarios with pre-rendered BEV video feeds.")
    else:
        renderable_options = filtered_df['scenario_id'].unique()

    if renderable_options:
        selected_scenario_id = st.selectbox(
            "Select Scenario ID to inspect (Rendered BEV Available):", 
            options=renderable_options
        )
        
        scene_info = df_summary[df_summary['scenario_id'] == selected_scenario_id].iloc[0] if selected_scenario_id in df_summary['scenario_id'].values else df_videos[df_videos['scenario_id'] == selected_scenario_id].iloc[0]
        
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
            
            video_url = f"{GCS_VIDEO_BASE_URL}/{selected_scenario_id}.mp4"
            
            col_vid, col_meta = st.columns([2, 1])
            
            with col_vid:
                st.video(video_url)
                st.caption(f"⚡ Streaming BEV video feed from GCS bucket for `{selected_scenario_id}`")
            
            with col_meta:
                st.markdown("### ⚠️ Scenario Risk Analysis")
                
                # 1. Dynamic Risk Level Badge
                risk_score = scene_info.get('predicted_risk_probability', 0)
                if risk_score >= 0.90:
                    st.error(f"🔴 **CRITICAL RISK** ({risk_score:.1%})")
                elif risk_score >= 0.80:
                    st.warning(f"🟠 **HIGH RISK** ({risk_score:.1%})")
                else:
                    st.info(f"🟡 **MODERATE RISK** ({risk_score:.1%})")

                # 2. Extract Key Telemetry
                max_decel = scene_info.get('max_deceleration', 0)
                max_vel = scene_info.get('max_velocity_mps', 0)
                ped_count = scene_info.get('pedestrian_count', 0)
                veh_count = scene_info.get('vehicle_count', 0)
                
                # 3. Determine Primary Risk Driver (XAI Logic)
                if max_decel < -5.0:
                    primary_driver = "🚨 Emergency Hard Braking"
                    explanation = f"Ego AV executed severe deceleration of `{max_decel:.1f} m/s²` to avoid collision."
                elif ped_count > 0 and risk_score > 0.85:
                    primary_driver = "🚸 Vulnerable Road User Threat"
                    explanation = f"High risk score driven by interaction with `{int(ped_count)}` pedestrian(s) in close proximity."
                elif max_vel > 18.0:
                    primary_driver = "⚡ High-Speed Corridor Conflict"
                    explanation = f"High-speed navigation (`{max_vel:.1f} m/s`) in dense surrounding traffic (`{int(veh_count)}` vehicles)."
                else:
                    primary_driver = "🚗 High Agent Density Intersection"
                    explanation = f"Complex multi-agent interaction involving `{int(veh_count)}` surrounding entities."

                st.markdown("**Primary Risk Driver:**")
                st.markdown(f"#### {primary_driver}")
                st.caption(explanation)
                
                st.divider()
                
                # 4. Telemetry Metrics
                st.markdown("### 📊 Scenario Telemetry")
                st.metric("Max Velocity", f"{max_vel:.1f} m/s")
                st.metric("Max Deceleration", f"{max_decel:.1f} m/s²", delta="Sudden Stop" if max_decel < -4.0 else None, delta_color="inverse")
                st.metric("Active Road Users", f"{int(veh_count + ped_count)} agents ({int(ped_count)} peds)")
                
    else:
        st.warning("No scenarios with rendered video feeds match the current filter criteria.")

except Exception as e:
    st.error(f"Error executing Streamlit dashboard: {e}")
