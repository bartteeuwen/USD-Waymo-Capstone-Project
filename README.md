# Autonomous Driving Hazard Detection & Risk Evaluation
**Author:** Bart Teeuwen  
**Course:** Capstone Project | University of San Diego  
**Standard:** APA 7th Edition Framework  
**Repository:** [USD-Waymo-Capstone-Project](https://github.com/bartteeuwen/USD-Waymo-Capstone-Project)  

---

## Executive Summary & White Paper Overview
This project presents an end-to-end data engineering and predictive modeling pipeline applied to the **Waymo Open Dataset**. It stream-parses uncompressed `.tfrecord` motion telemetry files from Google Cloud Storage, extracts kinematic agent interactions, audits data missingness and physical velocity constraints (v <= 38.0 m/s), and renders Explainable AI (XAI) risk overlays.

Machine learning baseline models (Random Forest, LightGBM, and XGBoost) evaluate scene hazard indices to predict safety-critical autonomous driving conditions.

---

## Repository Architecture

```text
USD-Waymo-Capstone-Project/
│
├── Main_Notebook.ipynb            <-- Complete Master White Paper Workflow
├── README.md                      <-- Project Documentation & Abstract
├── requirements.txt               <-- Python Dependencies
├── app.py                         <-- Interactive Streamlit App Dashboard
│
├── code_library/                  <-- Modular Code Base (PEP 8 & APA 7)
│   ├── Data_Preparation.ipynb     <-- Stream-parsing, GCS, protobuf, XAI renderer
│   ├── Data_Exploration.ipynb     <-- Quality audit, physics filters, EDA charts
│   └── Modeling.ipynb             <-- ML models (Random Forest, XGBoost, LightGBM)
│
├── data/                          <-- Processed CSV Datasets & Metadata
│   └── high_risk_scenarios_valid.csv
│
├── images/                        <-- Visual Artifacts & Plot Exports
│
└── other_materials/              <-- Serialized Model Artifacts
    ├── model_features.pkl
    └── waymo_rf_model.pkl
```

---

## Modular Code Library Navigation

| Notebook | Focus & Methodology | Primary Outputs |
| :--- | :--- | :--- |
| [**Data_Preparation.ipynb**](./code_library/Data_Preparation.ipynb) | Stream-parsing Waymo Protobufs, feature extraction, 91-frame XAI video engine | Processed telemetry, MP4 clips |
| [**Data_Exploration.ipynb**](./code_library/Data_Exploration.ipynb) | Missingness verification (0% missing), velocity threshold audits (v <= 38 m/s) | Distribution plots, EDA figures |
| [**Modeling.ipynb**](./code_library/Modeling.ipynb) | Train/test split, Random Forest / LightGBM baseline training, risk scoring | `waymo_rf_model.pkl` |
| [**Main_Notebook.ipynb**](./Main_Notebook.ipynb) | Full White Paper narrative integrating pipeline steps end-to-end | Master Capstone Workflow |

---

## Author Profile & Contact
* **Author:** Bart Teeuwen
* **GitHub:** [@bartteeuwen](https://github.com/bartteeuwen)
* **Project Repository:** [USD-Waymo-Capstone-Project](https://github.com/bartteeuwen/USD-Waymo-Capstone-Project)