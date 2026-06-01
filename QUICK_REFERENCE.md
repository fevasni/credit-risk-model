# 📋 Credit Risk Model — Quick Reference Guide

## 🎯 Project Overview
**Bati Bank BNPL Credit Scoring Platform** — 4-Week Intensive Development  
**Status:** Tasks 1–2 ✅ COMPLETE | Tasks 3–4 (Weeks 2–4) 📅 UPCOMING

---

## 📂 Key Documents & Their Purpose

### **Primary Report: CREDIT_RISK_MODEL_REPORT.md** ⭐
- **What:** Comprehensive 4-week project report summarizing Tasks 1–2 and roadmap for Weeks 2–4
- **Who:** Executive stakeholders, data scientists, regulators, project managers
- **Contains:** Business context, EDA findings, 5 critical insights, 3-week development roadmap
- **Must Read:** YES — This is the primary project document

### **Archived Report: PROGRESS_REPORT_Tasks_1_2.md**
- **What:** Extended timeline version (12-week) — retained for reference only
- **Status:** ⚠️ DEPRECATED — Superseded by 4-week report
- **Reason:** Project accelerated to 4-week sprint; 12-week version no longer relevant

### **README.md**
- **What:** Project setup guide with Basel II framework documentation
- **Contains:** Environment setup, dependencies, regulatory framework, literature references
- **Must Read:** Before running first commands

### **EDA Notebook: notebooks/eda.ipynb**
- **What:** Jupyter notebook with 8 visualizations and 23 executed cells
- **Contains:** Data profiling, distributions, correlations, outlier detection, insights
- **Figures:** 8 visualizations (histograms, KDE, bar charts, heatmap, box plots, outlier analysis)
- **Must Run:** For generating/validating visualizations

---

## 📊 EDA Visualizations Reference

| # | Figure | Cell | Key Finding | Action |
|---|--------|------|-------------|--------|
| 1 | **Histograms** | Cell #3 | Right-skew in Amount; single-value distributions | Log-transform Amount |
| 2 | **KDE Plots** | Cell #4 | Bimodal structure; extended right tail | Consider segmentation |
| 3 | **Summary Stats** | Cell #2 | Skewness > 2; FraudResult imbalance (0.2%) | Stratified sampling needed |
| 4 | **Categorical Bars** | Cell #5 | Pareto concentration across categories | Binary encode + group tail |
| 5 | **Correlation Heatmap** | Cell #6 | Amount↔Value r=0.9897; Fraud↔Amount r=0.56 | **Drop Value immediately** |
| 6 | **Missing Values** | Cell #7 | **ZERO missing values** | No imputation needed |
| 7 | **Box Plots** | Cell #8 | Outliers are legitimate high-value transactions | Use robust scaling |
| 8 | **Outlier Detection** | Cell #9–10 | 71.7% of extreme transactions are fraudulent | Create `is_extreme_amount` feature |

**All visualizations embedded in:** `notebooks/eda.ipynb`  
**Access:** Open notebook in Jupyter; scroll to respective cells

---

## 🔑 Critical Findings from EDA

### **Top 5 Insights Driving Feature Engineering:**

1. **Amount Drives Fraud Risk** ⭐⭐⭐
   - FraudResult↔Amount correlation: r = 0.56
   - 71.7% of extreme transactions (Z-score > 3σ) are fraudulent
   - **Action:** Engineer fraud-derived features (fraud_rate, fraud_recency)

2. **Extreme Categorical Concentration**
   - ProductCategory: 90% from 2 categories
   - ChannelId: 95% from 2 channels
   - ProviderId: 80% from 3 providers
   - **Action:** Strategic binning; separate minority segment analysis

3. **Perfect Data Quality**
   - Zero missing values across 95,493 records
   - Exceptional Xente data governance
   - **Action:** Skip imputation; focus on transformations

4. **Critical Multicollinearity**
   - Amount ↔ Value: r = 0.9897 (PERFECT)
   - **Action:** DROP VALUE COLUMN IMMEDIATELY

5. **Customer Diversity Enables RFM**
   - ~30,000 unique customers in 95,493 transactions
   - Rich data for aggregation
   - **Action:** Compute RFM vectors for clustering

---

## 📈 Project Timeline (4 Weeks)

### **Week 1 ✅ (COMPLETE)**
- ✅ Repository setup
- ✅ EDA executed (8 visualizations)
- ✅ Data quality assessment
- ✅ Comprehensive reports generated

### **Week 2 📅 (FEATURE ENGINEERING)**
- Feature engineering pipeline (`src/data_processing.py`)
- RFM aggregation
- WoE/IV transformation
- Categorical encoding
- **Success Criterion:** 20–30 engineered features ready

### **Week 3 📅 (PROXY TARGET & MODELING)**
- RFM clustering (K-Means)
- Proxy target validation
- Logistic regression training
- Benchmark models (Decision Tree, Random Forest)
- **Success Criterion:** Primary model AUC > 0.70

### **Week 4 📅 (DEPLOYMENT)**
- Model calibration & evaluation
- FastAPI service development
- Docker containerization
- CI/CD pipeline setup
- **Success Criterion:** Production-ready API (< 100ms latency)

---

## 🔧 Getting Started (Quick Start)

### **1. Clone & Setup**
```bash
cd credit-risk-model
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
```

### **2. View EDA Results**
```bash
jupyter notebook notebooks/eda.ipynb
# Scroll through cells 1–10 to see all visualizations
```

### **3. Read Reports**
- **Start Here:** `CREDIT_RISK_MODEL_REPORT.md` (4-week project report)
- **For Details:** `README.md` (setup & Basel II framework)
- **For Reference:** `PROGRESS_REPORT_Tasks_1_2.md` (archived 12-week version)

### **4. Review Data Dictionary**
```bash
head -20 data/Xente_Variable_Definitions.csv
```

---

## 📋 Feature Engineering Checklist (Week 2)

**Drop These:**
- [ ] ✂️ Value (multicollinear with Amount)
- [ ] ✂️ TransactionId, BatchId (unique identifiers)
- [ ] ✂️ SubscriptionId, CustomerId (redundant with AccountId)

**Transform These:**
- [ ] 🔄 Amount → log(Amount + 1)
- [ ] 🔄 Amount → quantile bins (for WoE)
- [ ] 🔄 TransactionStartTime → RFM features

**Create These:**
- [ ] ➕ Recency = days since last transaction
- [ ] ➕ Frequency = transaction count
- [ ] ➕ Monetary = total/average transaction value
- [ ] ➕ fraud_rate = fraud transactions / total transactions
- [ ] ➕ fraud_recency = days since last fraud
- [ ] ➕ is_extreme_amount = |Amount| > 3σ

**Encode These:**
- [ ] 🏷️ ProductCategory (binary top 2 + "Other")
- [ ] 🏷️ ChannelId (one-hot encode)
- [ ] 🏷️ PricingStrategy (one-hot encode)
- [ ] 🏷️ ProviderId (binary top provider + "Other")

---

## ⚠️ Known Constraints & Considerations

**Data Constraints:**
- Limited geographic diversity (96% single country)
- Severe fraud class imbalance (0.2%)
- Extreme Amount right-skew (need transformation)

**Regulatory Requirements:**
- Logistic Regression with WoE (interpretability mandate)
- Probability-based output (for Basel II capital calculations)
- Audit trails for all transformations

**Timeline Constraints:**
- 4 weeks total (compressed from typical 12-week projects)
- Weekly checkpoints required
- Prioritize model deployment over extensive experimentation

---

## 📞 Key Contacts & Escalation

**Project Lead:** [Name]  
**Data Science Lead:** [Name]  
**Regulatory Compliance:** [Name]  
**Bati Bank Sponsor:** [Name]

---

## 📚 Additional Resources

**Basel II Framework:**
- Basel Committee on Banking Supervision. (2005). "International Convergence of Capital Measurement and Capital Standards: A Revised Framework."

**Credit Scoring Methodologies:**
- Naeem Siddiqi. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*

**Python Libraries Used:**
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy (EDA & feature engineering)
- xverse (WoE/IV transformation)
- mlflow (experiment tracking)
- fastapi (model deployment)

---

**Last Updated:** May 31, 2026  
**Next Update:** End of Week 2 (Feature Engineering Checkpoint)
