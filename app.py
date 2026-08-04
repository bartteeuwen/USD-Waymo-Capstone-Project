import numpy as np
import pandas as pd
import streamlit as st

# Set Streamlit page config
st.set_page_config(
    page_title="Waymo AV Safety Triage Hub", page_icon="🚘", layout="wide"
)

# --- CONFIGURATION & PUBLIC GCS LINK ---
GCS_VIDEO_BASE_URL = (
    "https://storage.googleapis.com/waymo-capstone-rendered-videos"
)

# Exact Analytical Dataset Specs from Capstone Paper
TOTAL_EVALUATED_DATASET_SCENARIOS = 6311
CRITICAL_COMPLEXITY_COUNT = int(
    TOTAL_EVALUATED_DATASET_SCENARIOS * 0.41
)  # 2,588 (41%)
STANDARD_COMPLEXITY_COUNT = int(
    TOTAL_EVALUATED_DATASET_SCENARIOS * 0.59
)  # 3,723 (59%)


# --- DATA LOADING & FEATURE ENRICHMENT ---
@st.cache_data
def load_data():
  df = pd.read_csv("high_risk_scenarios_valid.csv")
  df["scenario_id"] = df["scenario_id"].astype(str).str.strip().str.lower()

  if "heuristic_risk_score" not in df.columns:
    df["heuristic_risk_score"] = np.clip(
        (np.abs(df["max_deceleration"]) / 10.0) * 0.4
        + (df["max_velocity_mps"] / 30.0) * 0.3
        + (df["predicted_risk_probability"] * 0.3),
        0.10,
        0.98,
    )

  def classify_road_context(row):
    vel = row.get("max_velocity_mps", 0)
    peds = row.get("pedestrian_count", 0)
    vehs = row.get("vehicle_count", 0)

    if vel > 15.0 and abs(row.get("max_deceleration", 0)) < 4.0:
      return "🛣️ High-Speed Corridor"
    elif peds > 0 or vehs > 8:
      return "🚥 Multi-Agent Intersection"
    else:
      return "🏙️ Urban Connector"

  df["road_context"] = df.apply(classify_road_context, axis=1)
  return df


try:
  df = load_data()
except Exception as e:
  st.error(f"Error loading dataset: {e}")
  st.stop()

# --- SIDEBAR NAVIGATION & FILTERS ---
st.sidebar.title("🚘 Waymo AV Safety Hub")

page = st.sidebar.radio(
    "Select View Mode:",
    ["📊 Executive Triage Dashboard", "Visual Inspection & Feedback"],
)

st.sidebar.divider()
st.sidebar.subheader("🎯 Triage Thresholds")

# Risk Probability Slider
min_risk_prob = st.sidebar.slider(
    "Minimum Model Risk Probability Score:",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
)

# Filter dataset based on active threshold
filtered_df = df[df["predicted_risk_probability"] >= min_risk_prob].copy()

if "feedback_db" not in st.session_state:
  st.session_state.feedback_db = {}


# ==============================================================================
# PAGE 1: EXECUTIVE TRIAGE DASHBOARD
# ==============================================================================
if page == "📊 Executive Triage Dashboard":
  st.title("🚘 Autonomous Vehicle Risk Triage & Productivity Engine")
  st.markdown(
      "Spatial-Temporal Modeling & Human-in-the-Loop Active Triage Pipeline"
  )
  st.divider()

  # --- TOP KPI ROW: EXACT DATASET DISTRIBUTION FROM CAPSTONE PAPER ---
  m1, m2, m3, m4 = st.columns(4)

  # Suppressed False Positives calculation comparing Heuristic vs Model predictions
  heuristic_high = len(df[df["heuristic_risk_score"] >= 0.70])
  model_high = len(df[df["predicted_risk_probability"] >= 0.70])
  false_positives_suppressed = max(0, heuristic_high - model_high)
  hours_saved = round((false_positives_suppressed * 5) / 60, 1)

  m1.metric("Total Evaluated Dataset", f"{TOTAL_EVALUATED_DATASET_SCENARIOS:,}")
  m2.metric("Critical Complexity (41%)", f"{CRITICAL_COMPLEXITY_COUNT:,}")
  m3.metric("Standard Complexity (59%)", f"{STANDARD_COMPLEXITY_COUNT:,}")
  m4.metric(
      "⏱️ Review Time Saved",
      f"{hours_saved} Hours",
      f"{false_positives_suppressed} False Alarms Suppressed",
  )

  st.divider()

  # --- SECTION: INTERACTIVE HIGH-RISK SCENARIOS TABLE ---
  st.subheader("📋 Triaged High-Risk Scenarios")
  st.caption(
      f"Showing **{len(filtered_df)}** scenarios meeting current threshold"
      f" (out of **{len(df)}** rendered inspection sample records)."
  )

  all_columns = [
      "scenario_id",
      "predicted_risk_probability",
      "heuristic_risk_score",
      "road_context",
      "max_velocity_mps",
      "max_deceleration",
      "vehicle_count",
      "pedestrian_count",
  ]

  default_cols = [
      "scenario_id",
      "predicted_risk_probability",
      "road_context",
      "max_velocity_mps",
      "max_deceleration",
      "vehicle_count",
  ]

  selected_cols = st.multiselect(
      "Choose Columns to Display:", options=all_columns, default=default_cols
  )

  search_query = st.text_input(
      "🔍 Search Scenario ID or Road Context:", placeholder="e.g. Corridor or 4b9..."
  )

  table_df = filtered_df.copy()
  if search_query:
    table_df = table_df[
        table_df["scenario_id"].str.contains(
            search_query.lower(), case=False, na=False
        )
        | table_df["road_context"].str.contains(
            search_query, case=False, na=False
        )
    ]

  if not selected_cols:
    selected_cols = default_cols

  st.dataframe(
      table_df[selected_cols].sort_values(
          by="predicted_risk_probability", ascending=False
      ),
      use_container_width=True,
      hide_index=True,
  )


# ==============================================================================
# PAGE 2: VISUAL INSPECTION & FEEDBACK
# ==============================================================================
elif page == "Visual Inspection & Feedback":
  st.title("Visual Inspection & Feedback Engine")
  st.markdown(
      "Trajectory playback, dual-validation metrics, and Safety Engineer feedback"
      " logging."
  )
  st.divider()

  total_rendered = len(df)
  flagged_scenarios = len(filtered_df)

  if filtered_df.empty:
    st.warning("No scenarios match the current threshold filter.")
  else:
    scenario_list = filtered_df["scenario_id"].unique().tolist()
    selected_sid = st.selectbox(
        f"Select Scenario ID to Inspect (Showing {flagged_scenarios} of"
        f" {total_rendered} sample scenarios):",
        options=scenario_list,
    )

    scene_info = filtered_df[
        filtered_df["scenario_id"] == selected_sid
    ].iloc[0]
    video_url = f"{GCS_VIDEO_BASE_URL}/{selected_sid}.mp4"

    col_video, col_meta = st.columns([1.3, 1.0])

    with col_video:
      st.subheader("Trajectory Stream")
      st.video(video_url, autoplay=True, loop=True)
      st.caption(
          "🔴 Waymo SDC | 🔵 Vehicle | 🟡 Pedestrian | 🟢 Cyclist | 💖 Risk"
          " Hazard Zone"
      )
      st.divider()

      st.subheader("📝 Safety Engineer Validation")
      with st.form(key=f"feedback_form_{selected_sid}"):
        is_true_risk = st.radio(
            "Validation Decision:",
            options=[
                "True High-Risk Event",
                "False Positive (Benign)",
                "Uncertain / Edge Case",
            ],
            horizontal=True,
        )
        engineer_notes = st.text_area(
            "Safety Engineer Notes:",
            placeholder="e.g. Turning vehicle proximity close but non-colliding...",
        )
        submit_btn = st.form_submit_button("Submit Feedback")

        if submit_btn:
          st.session_state.feedback_db[selected_sid] = {
              "validation": is_true_risk,
              "notes": engineer_notes,
          }
          st.success(f"✅ Feedback logged for `{selected_sid}`!")

      if selected_sid in st.session_state.feedback_db:
        logged = st.session_state.feedback_db[selected_sid]
        st.info(
            f"**Logged Status:** `{logged['validation']}` | Notes:"
            f" *{logged['notes']}*"
        )

    with col_meta:
      st.subheader("🛡️ Dual-Validation Analysis")

      model_score = scene_info.get("predicted_risk_probability", 0)
      heuristic_score = scene_info.get("heuristic_risk_score", 0)

      c_model, c_heur = st.columns(2)
      c_model.metric("Model Risk Score", f"{model_score:.1%}")
      c_heur.metric("Heuristic Score", f"{heuristic_score:.1%}")

      if heuristic_score > 0.70 and model_score < 0.40:
        st.info(
            "💡 **Filtered False Positive:** Rule heuristic flagged high"
            " velocity/proximity, but tree-based model confirmed safe trajectory"
            " separation."
        )
      elif model_score >= 0.80:
        st.error("🚨 **Validated High Risk:** High trajectory conflict probability.")
      else:
        st.warning("⚠️ **Moderate Risk:** Minor trajectory or velocity spike.")

      st.divider()

      max_decel = scene_info.get("max_deceleration", 0)
      max_vel = scene_info.get("max_velocity_mps", 0)
      ped_count = scene_info.get("pedestrian_count", 0)
      veh_count = scene_info.get("vehicle_count", 0)

      if max_decel < -4.5:
        primary_driver = "🚨 Emergency Hard Braking"
        explanation = (
            f"Waymo SDC deceleration of `{max_decel:.1f} m/s²` required to"
            " avoid conflict."
        )
      elif max_vel > 16.0:
        primary_driver = "⚡ High-Speed Corridor / Conflict"
        explanation = (
            f"High-speed navigation (`{max_vel * 2.237:.1f} mph`) through dense"
            " traffic."
        )
      elif ped_count > 0:
        primary_driver = "🚸 Vulnerable Road User Proximity"
        explanation = (
            f"Interaction with `{int(ped_count)}` pedestrian(s) near ego vehicle."
        )
      else:
        primary_driver = "🚗 Complex Multi-Agent Interaction"
        explanation = "High multi-agent density at trajectory intersection."

      st.markdown("**Primary Risk Driver:**")
      st.markdown(f"#### {primary_driver}")
      st.caption(explanation)

      st.divider()
      st.markdown("### 📊 Telemetry Summary")
      st.write(f"**Road Context:** {scene_info.get('road_context')}")
      st.metric("Max Velocity", f"{max_vel * 2.237:.1f} mph")
      st.metric("Max Deceleration", f"{max_decel:.1f} m/s²")
      st.metric("Surrounding Agents", f"{int(veh_count + ped_count)} total")
