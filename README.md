# 🔮 Customer Churn Prediction with Sentiment Signals

> **Multimodal ML:** CRM Tabular Features + NLP Sentiment from Support Tickets  
> Portfolio Project — Amazon ML Summer School | NIT Kurukshetra, AIML 2nd Year

---

## 📌 What This Project Does

Telecom companies lose revenue every time a customer cancels. This project builds a churn prediction system that goes beyond standard billing data by also analyzing **how frustrated customers sound in their support tickets**.

The core hypothesis: *a customer's sentiment in support interactions is a leading indicator of churn — and adding it to a model improves prediction.*

---

## 🏗️ Architecture

```
Telco CRM Data ──→ Tabular Feature Engineering ──┐
                                                   ├──→ Unified Features ──→ XGBoost / LightGBM
Support Tickets ──→ VADER Sentiment Extraction  ──┘
```

---

## 🔬 Ablation Study Design

Three models are trained and compared to isolate the exact contribution of sentiment features:

| Model | Features Used | Purpose |
|-------|--------------|---------|
| M1 — XGBoost Baseline | CRM tabular only | Control — no NLP |
| **M2 — XGBoost + Sentiment** | CRM + VADER scores | **Main hypothesis** |
| M3 — LightGBM + Sentiment | CRM + VADER scores | Alternative architecture |

---

## 📊 Results

| Model | AUC-ROC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| M1 — XGBoost (Tabular only) | 0.8389 | 0.6248 | 0.5379 | 0.7453 |
| **M2 — XGBoost + Sentiment** | **0.8658** | **0.6569** | **0.5763** | **0.7640** |
| M3 — LightGBM + Sentiment | 0.8618 | 0.6561 | 0.5693 | 0.7747 |

> **+2.69% AUC improvement** purely from adding NLP sentiment features.

---

## 💡 Key Findings

1. `avg_neg_score` and `sentiment_max` rank **3rd and 4th in SHAP importance** — beating MonthlyCharges
2. Churned customers have **2.4× higher negative ticket ratio** (72.9% vs 32.1%)
3. Mean VADER sentiment score: churned = **-0.290** vs retained = **+0.131**
4. Month-to-month contract customers churn at **42.7%** vs only **2.8%** for 2-year contracts

---

## 🗂️ Project Structure

```
customer-churn-prediction/
├── Customer_Churn_Prediction.ipynb   ← Main notebook (run top to bottom)
├── requirements.txt                  ← Python dependencies
├── outputs/
│   ├── eda_report.png                ← EDA visualizations
│   ├── sentiment_analysis.png        ← Sentiment distribution plots
│   ├── shap_feature_importance.png   ← SHAP feature importance
│   └── final_report.png              ← ROC curves + confusion matrix
└── data/
    └── raw/                          ← Auto-downloaded on first run
```

---

## ⚙️ How to Run

### Option 1: Google Colab (Recommended)
1. Open `Customer_Churn_Prediction.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Run all cells top to bottom (`Runtime → Run all`)
3. All dependencies install automatically in the first cell

### Option 2: Local
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
```

---

## 📦 Dependencies

```
xgboost
lightgbm
vaderSentiment
shap
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install all at once: `pip install -r requirements.txt`

---

## 📝 Data Sources & Disclaimer

**CRM Data:** [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d) — real, publicly available dataset (~7,000 customers).

**Support Tickets:** Synthetically generated for this project. Each ticket is created using a rule-based generator that assigns negative/neutral/positive language based on CRM risk signals (contract type, tenure, charges). The generator uses a fixed random seed (`seed=42`) for reproducibility. *Real-world deployment would require actual support ticket data.*

---

## 🧠 Why This Approach Is Interesting

- **Multimodal fusion** — combining structured (tabular) and unstructured (text) data is a real-world industry pattern used at scale
- **Rigorous ablation study** — isolates the exact contribution of each feature group rather than just reporting a final accuracy
- **SHAP explainability** — goes beyond accuracy to show *why* the model predicts what it does
- **Practical business framing** — each predicted churner = a retention campaign opportunity

---

## ⚠️ Limitations

- Support ticket data is synthetic; real ticket data may produce different (likely stronger) sentiment signals
- No hyperparameter tuning applied — default + light configurations used; Optuna/GridSearchCV could push AUC further
- VADER is a lexicon-based model and may miss sarcasm or domain-specific language; a fine-tuned transformer (e.g., DistilBERT) could improve sentiment quality
- Dataset is from a single telecom provider; generalization across industries is not guaranteed

---

## 👤 Author

**[Your Name]**  
B.Tech AIML, NIT Kurukshetra (2nd Year)  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)
