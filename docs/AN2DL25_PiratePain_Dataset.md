# AN2DL [2025-2026]: Time Series Classification
Time Series Classification: develop a deep learning model that classifies multivariate temporal series.

---

# AN2DL [2025-2026]: Time Series Classification

## Submit Prediction
## Dataset Description

### 🏴‍☠️ The Pirate Pain Dataset
Ahoy, matey! This dataset contains **multivariate time series** data, captured from both ordinary folk and pirates over repeated observations in time. Each sample collects temporal dynamics of body joints and pain perception, with the goal of predicting the subject’s true pain status:

- `no_pain`
- `low_pain`
- `high_pain`

### ⚓ Files
- `pirate_pain_train.csv` — training set  
- `pirate_pain_train_labels.csv` — labels for the training set  
- `pirate_pain_test.csv` — test set (with no labels)  
- `sample_submission.csv` — an example of random submission

### 🧭 Data Overview
Each record represents a **time step** within a subject’s recording, identified by `sample_index` and `time`. The dataset includes several groups of features:

- `pain_survey_1–pain_survey_4` — simple rule-based sensor aggregations estimating perceived pain.  
- `n_legs`, `n_hands`, `n_eyes` — subject characteristics.  
- `joint_00–joint_30` — continuous measurements of body joint angles (neck, elbow, knee, etc.) across time.

### 🏴‍☠️ Task
Predict the **real pain level** of each subject based on their **time-series motion data**.

---

## ⚙️ Data Loading
```python
import pandas as pd

X_train = pd.read_csv('pirate_pain_train.csv')
y_train = pd.read_csv('pirate_pain_train_labels.csv')

X_test = pd.read_csv('pirate_pain_test.csv')
```

---

## 🗺️ Validation
**No validation split be provided.** You’ll need to chart your own course and create one from the training data.

---

## ⬇️ Download via Kaggle CLI
```bash
kaggle competitions download -c an2dl2526c1
```