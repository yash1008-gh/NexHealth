NexHealth

NexHealth is an end-to-end machine learning system for diabetes readmission risk prediction. It includes leakage-safe grouped training, engineered clinical features, calibrated LightGBM inference, persisted decision thresholds, local SHAP explanations, a FastAPI backend, and a Streamlit frontend for interactive testing.

Dataset

This project uses the Diabetes 130-US Hospitals for Years 1999-2008 dataset from the UCI Machine Learning Repository. The dataset contains ten years of inpatient diabetes encounter records from 130 U.S. hospitals and integrated delivery networks.

Project Highlights
Leakage-safe train/validation/test splitting using grouped splits by patient.
Custom feature engineering for diagnosis grouping, utilization features, medication behavior, and clinical risk signals.
Calibrated LightGBM as the main model, with threshold tuning for decision-making.
Saved inference threshold in artifacts/threshold.json instead of using a hardcoded 0.5 cutoff.
RawInputAdapter to accept raw ID-based inputs and enrich them safely before feature engineering.
FastAPI serving layer with strict request validation using Pydantic.
Local per-patient SHAP explanations returned through the API.
Streamlit frontend for live testing with manual inputs and dummy low-risk / high-risk test patients.
Centralized logging for training and serving.
Model Performance

On the untouched test split, the calibrated LightGBM model achieved:

F2 Score: 0.6846
Precision: 0.5873
Recall: 0.7141
ROC-AUC: 0.7000
Selected Threshold: 0.430
Repository Structure
NexHealth/
├── api/
│   ├── app.py
│   └── schemas.py
├── artifacts/
│   ├── calibrated_pipeline.pkl
│   ├── explainer_pipeline.pkl
│   ├── preprocessor.pkl
│   ├── shap_summary.png
│   ├── shap_values.npy
│   └── threshold.json
├── Data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── src/
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── raw_input_adapter.py
│   └── train.py
├── Streamlit/
│   └── app.py
└── utils/
    ├── logger.py
    └── threshold.py
How It Works
Raw patient input is validated in the API.
The RawInputAdapter enriches raw ID fields into description columns.
NexHealthFeatureEngineer creates derived clinical features.
The pipeline preprocesses the data and sends it to the calibrated LightGBM model.
The saved threshold from threshold.json is used to decide the final class label.
If requested, the API returns local SHAP contributions for that specific patient.
Setup

Install the serving dependencies:

pip install -r requirements-serving.txt

If you want to retrain the model and refresh all artifacts:

python -m src.train
Run the FastAPI Server
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
Run the Streamlit App
streamlit run Streamlit/app.py
API Endpoints
GET /health

Returns basic readiness information and confirms the serving artifacts are loaded.

POST /predict?explain=true|false

Accepts a validated raw patient payload and returns:

predicted probability
binary prediction
threshold used
optional local SHAP contributions for that patient
Notes
calibrated_pipeline.pkl is the inference artifact.
explainer_pipeline.pkl is used for SHAP only.
preprocessor.pkl is kept for inspection/debugging and is not part of inference.
The saved threshold is used during inference, so predictions stay consistent with training.
