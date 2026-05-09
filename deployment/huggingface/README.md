---
title: Heart Disease API
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: ML-powered API for heart disease risk prediction
---

# Heart Disease Risk Prediction — API

ML-powered FastAPI service serving a Random Forest classifier
(ROC-AUC ~0.96) trained on the UCI Heart Disease dataset.

## Model selection

The training pipeline (`src/models/train.py` in the
[source repo](https://github.com/dharmendra-2025cs05041/heart-disease-mlops))
trains four candidate algorithms — Logistic Regression, Random Forest,
Gradient Boosting and SVM — each tuned with 5-fold `GridSearchCV`
(scoring = `roc_auc`). The candidate with the highest **test ROC-AUC**
on the held-out 20 % test split is byte-copied to `final_model.pkl`,
which is the only model file shipped to this Space.

Latest training run:

| Model               | Test ROC-AUC | Selected? |
|---------------------|-------------:|-----------|
| **Random Forest**   | **0.9610**   | ✅ promoted to `final_model.pkl` |
| Logistic Regression | 0.9578       |           |
| SVM                 | 0.9470       |           |
| Gradient Boosting   | 0.8983       |           |

If a future re-train flips the ranking, the deploy script
(`deployment/huggingface/deploy.sh`) automatically re-uploads whichever
estimator now lives in `final_model.pkl` — no code change required
on this Space. Hit `GET /info` to see which algorithm is currently
loaded.

## Endpoints

| Method | Path        | Purpose                                |
|--------|-------------|----------------------------------------|
| GET    | `/`         | Welcome message                                   |
| GET    | `/health`   | Liveness probe (reports `model_loaded`)           |
| GET    | `/info`     | Model type + 13-feature schema                    |
| GET    | `/docs`     | Interactive Swagger UI                            |
| POST   | `/predict`  | Heart disease risk prediction                     |
| GET    | `/metrics`  | Prometheus exposition format                      |

## Try it

Open the [Swagger UI](https://dharmendra-2025cs05041-heart-disease-api.hf.space/docs)
or use the curl samples below.

The Swagger UI ships with two convenience features for evaluators:

- A **Servers** dropdown at the top, pre-set to this public Space
  (with an alternate *Local* entry for self-hosters).
- An **Examples** dropdown on `POST /predict` with three labelled
  scenarios — 🔴 High-risk patient, 🟢 Low-risk patient,
  ❌ Invalid payload — so you can run a representative request
  without typing any JSON.

Permissive CORS is enabled, so the *Try it out* button also works when
the Swagger UI is opened from a different origin.

> The free CPU Space sleeps when idle, so the very first request after
> a period of inactivity takes ~30 s while the container cold-starts.
> Subsequent requests respond in well under 100 ms.

### 1) Positive case — high-risk patient (expects `prediction: 1`)

```bash
curl -X POST https://dharmendra-2025cs05041-heart-disease-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67, "sex": 1, "cp": 0, "trestbps": 160, "chol": 286,
    "fbs": 0, "restecg": 0, "thalach": 108, "exang": 1,
    "oldpeak": 1.5, "slope": 1, "ca": 3, "thal": 2
  }'
```

Sample response:

```json
{
  "prediction": 1,
  "probability_no_disease": 0.396,
  "probability_disease":   0.604,
  "confidence":            0.604,
  "risk_level": "Moderate"
}
```

### 2) Negative case — low-risk patient (expects `prediction: 0`)

```bash
curl -X POST https://dharmendra-2025cs05041-heart-disease-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 200,
    "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
    "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2
  }'
```

Sample response:

```json
{
  "prediction": 0,
  "probability_no_disease": 0.970,
  "probability_disease":   0.030,
  "confidence":            0.970,
  "risk_level": "Low"
}
```

### 3) Validation-error case — out-of-range fields (expects HTTP 422)

```bash
curl -X POST https://dharmendra-2025cs05041-heart-disease-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 250, "sex": 5, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

Sample response:

```json
{
  "detail": [
    {"loc":["body","age"],"msg":"Input should be less than or equal to 120","type":"less_than_equal"},
    {"loc":["body","sex"],"msg":"Input should be less than or equal to 1","type":"less_than_equal"}
  ]
}
```

### Feature reference

| Field      | Type  | Range / values                                  |
|------------|-------|-------------------------------------------------|
| `age`      | int   | 1–120 (years)                                   |
| `sex`      | int   | 0 = female, 1 = male                            |
| `cp`       | int   | 0–3 (chest-pain type)                           |
| `trestbps` | int   | 50–250 (resting BP, mm Hg)                      |
| `chol`     | int   | 100–600 (serum cholesterol, mg/dl)              |
| `fbs`      | int   | 0 or 1 (fasting blood sugar > 120 mg/dl)        |
| `restecg`  | int   | 0–2 (resting ECG result)                        |
| `thalach`  | int   | 60–250 (max heart rate achieved)                |
| `exang`    | int   | 0 or 1 (exercise-induced angina)                |
| `oldpeak`  | float | 0.0–10.0 (ST depression vs rest)                |
| `slope`    | int   | 0–2 (slope of peak exercise ST segment)         |
| `ca`       | int   | 0–4 (number of major vessels coloured)          |
| `thal`     | int   | 0–3 (thalassemia)                               |

## Source

Full source, training pipeline, monitoring stack, Kubernetes manifests
and CI/CD configuration:
<https://github.com/dharmendra-2025cs05041/heart-disease-mlops>

## Author

Dharmendra Parsaila (2025CS05041) — M.Tech, MLOps Assignment 1.
