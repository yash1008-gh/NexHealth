# NexHealth

NexHealth is an end-to-end machine learning system for diabetes readmission risk prediction.

It includes:

* leakage-safe grouped training by patient
* custom clinical feature engineering
* calibrated LightGBM inference
* saved decision thresholding
* local SHAP explanations for individual predictions
* a FastAPI serving layer
* a Streamlit frontend
* Dockerized deployment for local and server execution

---

## Live Demo

The project is deployed on an AWS EC2 `t3.micro` instance:

**[http://3.106.217.146/](http://3.106.217.146/)**

The instance is often stopped to save resources, so the demo may not always be reachable.

To keep the app usable on the small `t3.micro` instance, an additional **2 GB of memory / swap support** was allocated on the server side to help the containers run more reliably under memory pressure.

---

## Project Overview

This project predicts whether a diabetic patient is likely to be readmitted to the hospital.

The system is designed as a full pipeline, not just a standalone model:

1. Raw patient input is validated at the API layer.
2. Raw ID fields are enriched using the dataset mapping file.
3. Clinical and utilization features are engineered.
4. The processed record is passed through the trained calibrated model.
5. A saved threshold is used to convert probability into a final class label.
6. Optional SHAP explanations are returned for the specific patient.
7. The Streamlit app provides a simple UI for interactive testing.

---

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the UCI Machine Learning Repository.

The dataset contains inpatient encounter records from 130 U.S. hospitals and integrated delivery networks.

Supporting ID descriptions are loaded from `Data/IDS_mapping.csv`.

---

## Architecture

### 1. Training and preprocessing

The training pipeline lives in `src/` and is responsible for:

* loading the raw dataset
* applying eligibility filters
* preparing the binary target label
* splitting data by patient groups to avoid leakage
* engineering domain-specific features
* training and calibrating the classifier
* selecting the best decision threshold
* generating SHAP artifacts
* saving the final model artifacts

### 2. Serving layer

The FastAPI app in `api/` exposes:

* `GET /health`
* `POST /predict?explain=true|false`

This layer loads the saved artifacts and performs inference.

### 3. User interface

The Streamlit app in `Streamlit/` sends raw patient data to the API and displays:

* prediction
* probability
* threshold used
* SHAP feature impact chart
* SHAP feature contribution table

### 4. Dockerized deployment

The project is containerized with:

* `Dockerfile.api` for the FastAPI backend
* `Dockerfile.ui` for the Streamlit frontend
* `docker-compose.yml` to run both services together

The Streamlit container talks to the API container using the Docker network hostname `http://api:8000`.

---

## Repository Structure

```text
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
├── utils/
│   ├── logger.py
│   └── threshold.py
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements-serving.txt
└── README.md
```

---

## How the System Works

### Training flow

#### `src/preprocess.py`

This file handles raw dataset loading and preprocessing.

It:

* reads the CSV dataset
* merges ID description mappings
* removes invalid or excluded records
* converts the readmission target into binary form
* converts age and weight ranges into numeric midpoints where needed
* builds preprocessing pipelines for both one-hot and LightGBM-friendly flows

#### `src/feature_engineering.py`

This file creates the model’s domain features.

It transforms raw hospital fields into signals such as:

* diagnosis groupings
* service utilization
* medication intensity
* care intensity
* chronic burden
* insulin usage and medication change indicators
* relapse probability features
* high comorbidity indicators

It also compresses rare specialty values into `Other` and standardizes missing values.

#### `src/train.py`

This is the training orchestrator.

It:

* loads the raw data
* filters it
* builds leakage-safe train/validation/test splits using patient grouping
* trains baseline models and the final LightGBM pipeline
* calibrates predicted probabilities
* selects the best threshold
* saves artifacts to `artifacts/`
* generates SHAP outputs

The training code uses grouped splitting so the same patient does not appear in both training and evaluation. That is the bare minimum for not lying to yourself.

#### `src/predict.py`

This file defines the saved artifact paths and helper functions for loading the trained pipelines and threshold.

---

### Inference flow

#### `api/schemas.py`

Defines the request and response schema for the API.

It includes strict field validation for:

* demographic data
* admission details
* laboratory buckets
* diagnosis fields
* medication statuses
* binary flags

Unexpected extra fields are rejected.

#### `src/raw_input_adapter.py`

The raw API input uses ID-based values, so this adapter enriches them using the ID mapping file before feature engineering.

#### `api/app.py`

The FastAPI app performs the following steps when a prediction request arrives:

1. validate the payload
2. convert the request into a one-row DataFrame
3. load the calibrated pipeline
4. compute probability with `predict_proba()`
5. compare probability against the saved threshold
6. return the prediction and probability
7. optionally return local SHAP values for the patient

The `/health` endpoint is used to confirm that the serving artifacts are loaded.

#### `Streamlit/app.py`

The UI collects patient features, sends them to the API, and displays the result.

It also includes quick test presets for:

* low risk patient
* high risk patient

This makes it easy to test the system without manually filling every field.

---

## Dockerized Setup

### API container

`Dockerfile.api` builds the FastAPI backend and exposes port `8000`.

### UI container

`Dockerfile.ui` builds the Streamlit frontend and exposes port `8501`.

### Compose orchestration

`docker-compose.yml` runs both containers on the same bridge network.

Important detail:

* the UI container uses `API_URL=http://api:8000`
* the hostname `api` works because Docker Compose creates service-name DNS inside the shared network

That means the frontend can talk to the backend internally without hardcoding `localhost`, which would be wrong inside containers.

---

## Requirements

For serving and inference:

```bash
pip install -r requirements-serving.txt
```

Main packages include:

* FastAPI
* Streamlit
* Requests
* Uvicorn
* SHAP
* Pydantic
* LightGBM
* scikit-learn

---

## Running Locally

### With Docker

Build and run the stack:

```bash
docker compose up --build
```

Then open:

* Streamlit UI: `http://localhost:8501`
* FastAPI docs: `http://localhost:8000/docs`

### Without Docker

Run the backend:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Run the Streamlit app:

```bash
streamlit run Streamlit/app.py
```

If the frontend cannot reach the API, set `API_URL` appropriately.

---

## API Endpoints

### `GET /health`

Returns readiness information and checks whether the model artifacts are loaded.

### `POST /predict?explain=true|false`

Accepts a validated patient payload and returns:

* predicted probability
* final binary prediction
* threshold used
* optional SHAP contributions

---

## Saved Artifacts

The repository stores the main inference artifacts in `artifacts/`:

* `calibrated_pipeline.pkl`
* `explainer_pipeline.pkl`
* `preprocessor.pkl`
* `shap_values.npy`
* `shap_summary.png`
* `threshold.json`

Notes:

* `calibrated_pipeline.pkl` is the main inference artifact
* `explainer_pipeline.pkl` is used for SHAP explanations
* `preprocessor.pkl` is kept for inspection and debugging
* `threshold.json` stores the chosen decision threshold so inference stays consistent with training

---

## Model Performance

On the untouched test split, the calibrated LightGBM model achieved:

* **F2 Score:** 0.6846
* **Precision:** 0.5873
* **Recall:** 0.7141
* **ROC-AUC:** 0.7000
* **Selected Threshold:** 0.430

---

## Notes on Deployment

The deployed version runs on an **AWS EC2 `t3.micro`** instance.

Because `t3.micro` has limited memory, the server was configured with **2 GB memory/swap support** to help the containers run more reliably.

The live link is:

**[http://54.153.223.224](http://54.153.223.224)**

The instance is frequently stopped to save cost, so the app may not always be available.

---

## Why This Design Works

This project is built the right way for a production-style ML demo:

* preprocessing is separated from serving
* raw inputs are validated before inference
* patient leakage is handled during training
* the decision threshold is saved and reused
* explanations are generated per patient
* Docker makes the app portable
* Streamlit stays decoupled from the model server

That is the architecture. Not glamorous, but functional. Which is more than most projects can claim.

---

## License

No license has been specified yet.
