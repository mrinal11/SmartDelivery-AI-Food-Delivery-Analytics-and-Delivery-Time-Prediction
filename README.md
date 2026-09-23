# SmartDelivery AI
## Food Delivery Analytics and Delivery Time Prediction

> **IBM SkillsBuild Data Analytics with AI Academic Internship**  
> Conducted by **BharatCares** in association with **AICTE**  
> Submitted by: **Mrinal Singh**

---

## Project Overview

SmartDelivery AI is a complete, end-to-end Data Analytics and Machine Learning project that analyzes real-world food delivery data to:

- Understand the operational factors that influence delivery time
- Engineer meaningful features from raw GPS, time, and categorical data
- Train and compare four supervised regression models
- Predict food delivery time with high accuracy (MAE ≈ 3.67 minutes, R² ≈ 0.746)
- Generate actionable business insights for operations managers

**ML Task:** Supervised Regression  
**Target Variable:** `Time_taken(min)` — delivery time in minutes  
**Best Model:** Random Forest Regressor

---

## Repository Structure

```
.
├── MrinalSingh_SmartDeliveryAI.ipynb   # Main Jupyter Notebook (59 cells)
├── MrinalSingh_ProjectReport.docx      # Full academic project report
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── updated.csv                         # Dataset (2,442 records, 20 columns)
└── smartdelivery_best_model.pkl        # Saved best model (after running notebook)
```

---

## Dataset

**File:** `updated.csv`  
**Records:** 2,442 rows × 20 columns  
**Source:** Food delivery operations data across multiple Indian cities

### Key Columns

| Column | Description |
|--------|-------------|
| `Delivery_person_Age` | Age of the delivery person |
| `Delivery_person_Ratings` | Delivery agent rating (1–5 scale) |
| `Restaurant_latitude/longitude` | GPS coordinates of restaurant |
| `Delivery_location_latitude/longitude` | GPS coordinates of delivery address |
| `Time_Orderd` | Time order was placed |
| `Time_Order_picked` | Time order was picked up |
| `Weatherconditions` | Encoded weather condition (0=Sunny ... 7=Tornado) |
| `Road_traffic_density` | Encoded traffic level (0=Low ... 5=Jam) |
| `multiple_deliveries` | Number of simultaneous deliveries (0–3) |
| `Festival` | Festival indicator (1=No, 2=Yes) |
| `City` | City type (1=Metropolitan ... 4=Rural) |
| `Time_taken(min)` | **TARGET** — delivery time in minutes |

---

## Setup Instructions

### 1. Clone / Download the project

```bash
git clone <repository-url>
cd smartdelivery-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook MrinalSingh_SmartDeliveryAI.ipynb
```

Or run all cells via nbconvert:

```bash
jupyter nbconvert --to notebook --execute --inplace MrinalSingh_SmartDeliveryAI.ipynb
```

> Make sure `updated.csv` is in the **same directory** as the notebook.

---

## Notebook Structure (59 cells)

| Section | Content |
|---------|---------|
| **Section 1** | Project Introduction and Problem Statement |
| **Section 2** | Library Imports |
| **Section 3** | Load Dataset — shape, types, statistics |
| **Section 4** | Data Quality Analysis — missing values, duplicates, outliers |
| **Section 5** | Data Cleaning — systematic, justified, reported |
| **Section 6** | Feature Engineering — Haversine distance, order hour, pickup delay |
| **Section 7** | EDA & Visualization — 11 analytical plots |
| **Section 8** | Feature Preparation and Train/Test Split |
| **Section 9** | Model Training — 4 regression models |
| **Section 10** | Model Evaluation — MAE, RMSE, R², Cross-Validation |
| **Section 11** | Save Best Model |
| **Section 12** | Business Insights and Recommendations |
| **Section 13** | Conclusions |

---

## Model Performance Summary

| Model | Test MAE (min) | Test RMSE (min) | Test R² |
|-------|---------------|----------------|---------|
| **Random Forest** ⭐ | **3.6663** | **4.6974** | **0.7459** |
| HistGradientBoosting | 3.6663 | 4.7960 | 0.7351 |
| Gradient Boosting | 3.6979 | 4.7506 | 0.7401 |
| Linear Regression | 5.5488 | 7.0103 | 0.4340 |

**5-Fold Cross-Validation (Random Forest):**  
CV MAE = 3.44 ± 0.05 min — highly stable, no overfitting detected.

---

## Key Feature Importances (Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `Delivery_person_Ratings` | 25.65% |
| 2 | `distance_km` (Haversine) | 14.40% |
| 3 | `Weatherconditions` | 12.00% |
| 4 | `multiple_deliveries` | 11.37% |
| 5 | `Road_traffic_density` | 9.95% |

---

## Business Insights (Summary)

1. **Traffic:** Jam conditions add measurable delay → integrate real-time traffic routing
2. **Batching:** Multiple simultaneous deliveries increase per-order time → limit to 1 for premium orders
3. **Festivals:** Festival periods show elevated delivery times → pre-position agents
4. **Agent Performance:** Higher-rated agents deliver faster → performance-based routing
5. **City Type:** Rural areas have longer delivery distances → deploy micro-fulfillment centres
6. **Peak Hours:** 18:00–21:00 is the slowest period → scale fleet during evening peak

---

## Data Cleaning Summary

| Issue | Count | Action |
|-------|-------|--------|
| Age = 0 | 93 rows | Dropped |
| Rating > 5.0 | 5 rows | Dropped |
| Zero GPS coordinates | 176 rows | Dropped |
| Missing Ratings | 94 rows | Imputed (median = 4.7) |
| Missing multiple_deliveries | 54 rows | Imputed (mode = 0) |
| Missing Time_Orderd | 89 rows | Dropped (required for feature) |

**Result:** 2,442 → 2,176 rows (10.9% removed)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| pandas | Data manipulation |
| numpy | Numerical computing |
| matplotlib + seaborn | Visualization |
| scikit-learn | ML models, pipelines |
| joblib | Model serialization |
| Jupyter Notebook | Interactive analysis |

---

## Author

**Mrinal Singh**  
IBM SkillsBuild Data Analytics with AI Academic Internship  
BharatCares × AICTE

---

*Project: SmartDelivery AI — Food Delivery Analytics and Delivery Time Prediction*
