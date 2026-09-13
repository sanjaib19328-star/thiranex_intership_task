# Task 2: Heart Disease Prediction & Diagnostic Classification

> **Thiranex Internship Task 2 — Machine Learning Pipeline**  
> **Dataset**: `dataset_heart.csv` (Statlog / Cleveland Heart Disease)  
> **Notebook**: [`Heart_Disease_Prediction.ipynb`](Heart_Disease_Prediction.ipynb)  
> **Environment**: Python 3.10+ / Scikit-Learn 1.3+

---

## 1. Project Overview & Clinical Problem Statement

Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated **17.9 million lives each year** according to the World Health Organization (WHO). Over four out of five CVD deaths are due to heart attacks and strokes, with one-third occurring prematurely in individuals under 70 years of age.

Early diagnosis, risk stratification, and timely medical intervention are vital for preventing adverse cardiac events. This project builds a clinical machine learning pipeline to classify whether a patient has heart disease based on clinical, demographic, and non-invasive diagnostic test results.

### Core Objectives:
1. Conduct thorough exploratory data analysis (EDA) to uncover physiological drivers of coronary heart disease.
2. Build an **architecturally leak-free** preprocessing pipeline using Scikit-Learn `Pipeline` and `ColumnTransformer`.
3. Appropriately handle continuous vs. categorical features (using **One-Hot Encoding** rather than imposing artificial ordinal distances on nominal categories).
4. Train and benchmark three distinct machine learning paradigms:
   - **Baseline Linear Model**: Regularized Logistic Regression (L2 penalty)
   - **Interpretable Tree Model**: Decision Tree Classifier (with explicit structural regularization)
   - **Ensemble Model**: Random Forest Classifier (Bootstrap Aggregation + Random Subspace Sampling)
5. Evaluate models using a clinical diagnostic framework prioritizing **Recall (Sensitivity)** and **ROC-AUC** alongside Accuracy and Precision to guard against fatal False Negatives.

---

## 2. Dataset Architecture & Clinical Feature Dictionary

The analysis is performed on the benchmark **Statlog Heart Disease dataset** (`dataset_heart.csv`), consisting of **270 patient records** and **13 diagnostic attributes**.

| # | Attribute | Type | Scale / Values | Clinical Description |
|---|---|---|---|---|
| 1 | `age` | Continuous | 29 – 77 years | Age of the patient in years |
| 2 | `sex` | Categorical | 0 = Female, 1 = Male | Biological sex |
| 3 | `chest pain type` | Categorical | 1, 2, 3, 4 | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| 4 | `resting blood pressure` | Continuous | 94 – 200 mm Hg | Resting systolic blood pressure on hospital admission |
| 5 | `serum cholestoral` | Continuous | 126 – 564 mg/dl | Serum cholesterol in mg/dl |
| 6 | `fasting blood sugar` | Binary | 0 = False, 1 = True | Fasting blood sugar > 120 mg/dl |
| 7 | `resting electrocardiographic results` | Categorical | 0, 1, 2 | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| 8 | `max heart rate` | Continuous | 71 – 202 bpm | Maximum heart rate achieved during exercise stress test |
| 9 | `exercise induced angina` | Binary | 0 = No, 1 = Yes | Presence of angina pectoris provoked by physical exertion |
| 10 | `oldpeak` | Continuous | 0.0 – 6.2 mm | ST depression induced by exercise relative to rest |
| 11 | `ST segment` | Categorical | 1, 2, 3 | Slope of peak exercise ST segment (1: Upsloping, 2: Flat, 3: Downsloping) |
| 12 | `major vessels` | Discrete | 0, 1, 2, 3 | Number of major coronary vessels colored by fluoroscopy |
| 13 | `thal` | Categorical | 3, 6, 7 | Thallium scintigraphy stress defect (3: Normal, 6: Fixed, 7: Reversible) |
| 14 | **`heart disease`** | **Target** | **1, 2 &rarr; 0, 1** | **Diagnosis: 0 = Absence (Healthy), 1 = Presence (Heart Disease)** |

---

## 3. Data Preprocessing & Leakage Prevention

### 3.1 Header Sanitization
The raw CSV contains a trailing space in the second header (`'sex '`). Column headers are trimmed (`df.columns = df.columns.str.strip()`) to guarantee robust property referencing.

### 3.2 Target Remapping
The raw dataset encodes the target as `1` (Absence) and `2` (Presence). To comply with standard binary classification standards and Scikit-Learn loss functions:
$$\text{heart disease} = 
\begin{cases} 
0 & \text{if raw value} = 1 \quad (\text{Absence / Healthy}, N=150, 55.6\%) \\
1 & \text{if raw value} = 2 \quad (\text{Presence / Heart Disease}, N=120, 44.4\%)
\end{cases}$$

### 3.3 Categorical Encoding & Clinical Rationale
Features like `chest pain type` (1–4), `thal` (3, 6, 7), `resting ecg` (0–2), and `ST segment` (1–3) represent discrete diagnostic categories. Treating them as continuous integers would impose an erroneous metric distance (e.g., assuming type 4 chest pain is "4 times greater" than type 1).  
We apply **One-Hot Encoding** (`OneHotEncoder(drop='first', handle_unknown='ignore')`), producing non-collinear indicator features without dimensionality explosion on 270 records.

### 3.4 Data Leakage Prevention via Scikit-Learn Pipelines
- **Continuous Features**: Scaled using `StandardScaler()` inside a `ColumnTransformer`.
- **Categorical Features**: Encoded using `OneHotEncoder(drop='first')` inside the same transformer.
- Both transformations are fitted **exclusively on the training split** via Scikit-Learn `Pipeline`, guaranteeing zero test set contamination.
- For tree-based models, continuous features use `passthrough` (as tree splits are invariant to monotonic scaling).

### 3.5 Stratified 80/20 Partitioning
- **Training Set**: 216 records (80%)
- **Test Set**: 54 records (20%)
- Split parameter: `stratify=y, random_state=42` to ensure identical class proportions (~55.6% healthy, ~44.4% disease) in both partitions.

---

## 4. Models & Methodological Progression

```text
                               Raw Data (270 rows)
                                       │
                                       ▼
                       Column Sanitization & Target Remap
                                       │
                                       ▼
                             Stratified 80/20 Split
                           (Train: 216, Test: 54)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         Numerical Features (5)                Categorical Features (8)
         ['age', 'bp', 'chol', 'hr', 'oldpeak'] ['sex', 'cp', 'fbs', 'ecg', ...]
                    │                                     │
             StandardScaler                         OneHotEncoder
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                             Scikit-Learn Pipelines
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        Logistic Regression      Decision Tree          Random Forest
        (L2, max_iter=1000)   (max_depth=4, split=10)   (100 estimators)
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       ▼
                          Held-Out Test Set Evaluation
                      (Accuracy, Recall, ROC-AUC, CM, MDI)
```

---

## 5. Experimental Results & Benchmark

All models were evaluated on the **untouched test set ($N=54$, 30 Absence, 24 Presence)**:

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | False Negatives (Type II) | False Positives (Type I) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **87.04%** | **81.48%** | **91.67%** | **86.27%** | **0.9111** | **2 / 24** | **5 / 30** |
| **Random Forest** | **83.33%** | **80.00%** | **83.33%** | **81.63%** | **0.8938** | **4 / 24** | **5 / 30** |
| **Decision Tree** | **72.22%** | **68.00%** | **70.83%** | **69.39%** | **0.8222** | **7 / 24** | **8 / 30** |

---

## 6. Key Clinical Findings & Explainability

### 6.1 Why Logistic Regression is the Best Candidate
In clinical diagnostic screening, **Recall (Sensitivity)** is of paramount importance:
- A **False Positive** leads to non-invasive secondary confirmatory testing (e.g., stress echocardiography or CT coronary angiography).
- A **False Negative** sends a patient with active ischemic coronary disease home without treatment, creating a critical risk of acute myocardial infarction or cardiac arrest.

**Logistic Regression achieved 91.67% Recall**, missing only **2 out of 24 diseased patients** on the test set, alongside the highest overall Accuracy (87.04%) and ROC-AUC (0.9111). On low-sample tabular medical datasets ($N=270$), regularized linear models often outperform unconstrained tree algorithms because their constrained hypothesis space avoids fitting idiosyncratic sample noise.

### 6.2 Top Clinical Risk Factors (Random Forest MDI Feature Importance)
1. **`max heart rate` (13.0%)**: Chronotropic incompetence (failure to achieve expected peak exercise heart rate) reflects reduced cardiac output and coronary reserve.
2. **`thal_7` Reversible Defect (11.3%)**: Represents transient myocardial ischemia where myocardial perfusion is compromised during exertion but normalizes at rest—a definitive diagnostic indicator of coronary stenosis.
3. **`oldpeak` (10.2%)**: Exercise-induced ST depression reflects the clinical magnitude of myocardial ischemia under metabolic stress.
4. **`age` (9.4%)**: Independent biological risk factor tracking progressive coronary calcification.
5. **`chest pain type_4` Asymptomatic (9.0%)**: Patients with severe silent ischemia often present with advanced multivessel disease because they lack warning angina.

---

## 7. Project Structure

```text
Task-2/
├── dataset_heart.csv                # Original Statlog/Cleveland heart dataset (270 rows)
├── Heart_Disease_Prediction.ipynb   # Complete, executed 20-section Jupyter Notebook
├── README.md                        # Comprehensive technical documentation & report
└── requirements.txt                 # Pinned dependencies for full reproducibility
```

---

## 8. Installation & Reproduction Guide

### Step 1: Clone or Navigate to Directory
```bash
cd Task-2
```

### Step 2: Install Dependencies
Ensure Python 3.10+ is installed, then run:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Jupyter Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter notebook Heart_Disease_Prediction.ipynb
```
Or open [`Heart_Disease_Prediction.ipynb`](Heart_Disease_Prediction.ipynb) directly inside VS Code with the Python / Jupyter extension. All cells are already fully executed with embedded plots and outputs.

---

## 9. Clinical Limitations & Future Work

1. **Sample Size Limitations**: With $N=270$, small patient subsets (e.g., rare ECG abnormalities) have low representation. Validation on larger EHR cohorts ($N > 10,000$) is recommended.
2. **Biomarker Modernization**: The dataset lacks contemporary cardiac biomarkers (high-sensitivity troponin $hs\text{-cTnI/T}$, $NT\text{-proBNP}$, and coronary calcium score). Integrating these would enhance diagnostic specificity.
3. **Clinical Decision Support Role**: This machine learning system is engineered as an auxiliary screening and triage tool to assist cardiologists, not as an autonomous diagnostic authority.
