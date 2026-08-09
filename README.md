# Turning Autonomous Driving Motion Data into Interpretable Scene Intelligence

**Author**: Bart Sosa-Teeuwen  
**Program**: Master of Science in Applied Data Science | Shiley-Marcos School of Engineering | University of San Diego  
**Contact**: bteeuwen@sandiego.edu  

---

## 📌 Executive Summary

Autonomous vehicle (AV) fleets generate petabytes of raw movement logs, creating a massive computational bottleneck for safety teams seeking to isolate high-risk driving events. Current workflows rely on either rigid, oversimplified heuristic rules or compute-heavy, opaque deep-learning models.

This project delivers an auditable, lightweight diagnostic framework that combines automated spatial feature engineering with transparent tree-based models. By mapping raw vehicle paths and static map infrastructure ("map friction") into interpretable risk metrics, the system filters routine driving hours and flags high-complexity scenarios in under **15 ms** per scene.

```
┌────────────────────────┐      ┌──────────────────────────┐      ┌───────────────────────────┐
│ Raw Waymo Telemetry    │ ───► │ Feature Engineering      │ ───► │ Transparent Tree Model    │
│ (10-Hz Motion Tracks)  │      │ (Map Friction & Dynamics)│      │ (Sub-15ms Scenario Risk)  │
└────────────────────────┘      └──────────────────────────┘      └───────────────────────────┘
│
▼
┌───────────────────────────┐
│ Interactive Streamlit     │
│ HITL Review Dashboard     │
└───────────────────────────┘
```

---

## 🎯 Business & Strategic Objectives

This capstone addresses three critical organizational pillars within the AV industry:

* **Capture Market Share**: Accelerates commercial AV deployment by automating log triage, allowing safety teams to spend less time manually digging through routine drives and more time resolving critical edge cases.
* **Preserve Capital**: Replaces expensive, uninterpretable deep-learning re-simulations with lightweight, transparent data pipelines that reduce cloud infrastructure costs.
* **Demonstrate Trust**: Replaces "black-box" model outputs with explicit, rule-backed safety metrics and SHAP explanations for regulatory auditors and local communities.

---

## 📊 Dataset & Feature Architecture

### Data Ingestion & Cleaning
* **Source Data**: Waymo Open Motion Dataset (v1.2.0), streaming a 10% randomized sample from Google Cloud Storage (GCS) across 6 U.S. cities.
* **Analytical Sample**: **6,311** clean, 20-second driving scenarios (extracted from an initial 6,530 unique records).
* **Physical Anomaly Screening**: Applied a peak physical velocity threshold of **38.0 m/s** (85 mph) to eliminate GPS drift, coordinate jumps, and sensor dropouts.

### Feature Schema
Features are engineered across three geometric and dynamic categories:

1. **Movement Summaries**: Peak scalar velocity ($v_{max}$) and peak deceleration bounds across 20-second windows.
2. **Map Friction Identifiers**: Static infrastructure counts inside the scene bounding box (`crosswalk_count`, `stop_sign_count`, `speed_bump_count`, `lane_count`).
3. **Dynamic Interaction Metrics**: Minimum inter-agent spatial proximity (`min_inter_agent_dist`), mean agent velocity (`avg_agent_velocity`), and velocity standard deviation (`velocity_std`).

### Target Variable
* **`target_risk_matrix`**: Binary label derived from kinematic clustering (**1** = Critical Complexity [41%], **0** = Standard Complexity [59%]).

---

## 🧪 Modeling & Performance Benchmarks

All models were evaluated using an **80/20** stratified holdout split (**5,048** training scenarios / **1,263** test scenarios).

### Benchmark Comparison

| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Primary Operational Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Majority Baseline** | 58.67% | 0.0000 | 0.0000 | 0.0000 | 0.5000 | Performance floor baseline |
| **Logistic Regression** | 68.33% | 0.6667 | 0.4674 | 0.5495 | 0.7390 | Linear separability baseline |
| **Random Forest (Production)** | **69.28%** | **0.6402** | **0.5862** | **0.6120** | **0.7529** | **Sub-15 ms inference, explicit SHAP interpretability** |
| **LightGBM** | 68.96% | 0.6337 | 0.5900 | 0.6111 | 0.7548 | Gradient boosted baseline |
| **XGBoost** | 69.20% | 0.6383 | 0.5881 | 0.6122 | 0.7558 | High discriminatory performance |
| **Spatial GNN (GCN)** | 58.67% | 0.0000 | 0.0000 | 0.0000 | 0.5696 | Scene graph baseline (node feature averaging) |
| **Graph Attention Network (GAT)** | **72.05%** | — | — | — | — | Highest raw accuracy via multi-head attention |

> **Model Selection Decision**: While the Graph Attention Network (GAT) achieved the highest overall accuracy (**72.05%**), the lightweight **Random Forest** model (**69.28%** accuracy, **0.7529** ROC-AUC) was chosen for deployment. It delivers sub-**15 ms** execution times and provides transparent SHAP decision logic without requiring specialized GPU hardware.

---

## 🔎 Interpretability & Key Findings

* **Speed Variance Drives Risk**: Global SHAP analysis revealed that speed variation among surrounding actors (`velocity_std`) and minimum spatial proximity (`min_inter_agent_dist`) exert stronger pushes on scene complexity than raw vehicle volume alone.
* **Non-Linear Friction**: Pedestrian density non-linearly peaks in areas with moderate structural complexity (10 to 25 traffic controls/crosswalks) rather than massive multi-lane highway geometries.

---

## 🛠️ Project Artifact & Web Dashboard

The analytical core is operationalized as an interactive **Streamlit** executive dashboard and inspection tool (`app.py`):

1. **Dynamic Risk Triage**: Slide risk probability thresholds (e.g., set to 0.25) to filter workloads, quantify false alarm suppression, and calculate engineer hours saved.
2. **Live Custom Scenario Sensitivity**: Input custom map friction and motion parameters to execute real-time risk scoring (< **15 ms** latency).
3. **Trajectory Playback**: Stream MP4 motion trajectory renders hosted on Google Cloud Storage (GCS) alongside dual-validation metrics.
4. **Human-in-the-Loop Logging**: Record safety engineer overrides into session state for active learning pipeline retraining.

---

## 📁 Repository Structure

```text
.
├── artifacts/
│   ├── rf_model.pkl                    # Serialized Random Forest model
│   ├── kmeans_target.pkl               # Fitted KMeans model for target labeling
│   └── model_features.pkl              # Serialized feature list
├── data/
│   └── high_risk_scenarios_valid.csv   # Pre-scored validation dataset
├── notebook/
│   └── capstone_whitepaper_notebook.ipynb # Executive analysis & diagnostic framework
├── src/
│   ├── config.py                       # Project constants and feature lists
│   ├── data_loader.py                  # TFRecord scenario extraction pipeline
│   ├── feature_engineering.py          # Spatial joins & kinematic feature generation
│   ├── inference.py                    # Lightweight prediction engine for web app
│   ├── modeling.py                     # Tabular & Graph neural network architectures
│   └── tuning.py                       # RandomizedSearchCV & GNN hyperparameter search
├── app.py                              # Streamlit web dashboard application
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
* Python 3.10 or higher
* Pip environment manager

### 2. Setup Environment
```bash
# Clone repository
git clone [https://github.com/bartteeuwen/USD-Waymo-Capstone-Project.git](https://github.com/bartteeuwen/USD-Waymo-Capstone-Project.git)
cd USD-Waymo-Capstone-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Interactive Artifact
```bash
streamlit run app.py
```

---

## 📚 References & Acknowledgments

* **Data Source**: Waymo Open Motion Dataset (v1.2.0).
* **Key Literature**: Ettinger et al. (2021), Ki et al. (2025), Breiman (2001).
* *See full capstone report for complete bibliography and methodology notes.*
