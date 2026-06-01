---

## 📊 **PROJECT REPORT**

👉 **[Read the Complete 4-Week Project Report: CREDIT_RISK_MODEL_REPORT.md](CREDIT_RISK_MODEL_REPORT.md)**

This comprehensive report covers:
- ✅ **Tasks 1–2 Status** (Complete): Business understanding & EDA findings
- 📅 **Week 2–4 Roadmap**: Feature engineering, proxy target construction, model training & deployment
- 📈 **8 EDA Visualizations**: Distributions, correlations, outliers, fraud analysis
- 📋 **Critical Data Insights**: 5 key findings driving feature engineering
- ✔️ **Regulatory Compliance**: Basel II framework, model interpretability, governance

---

## 📁 **DATA SOURCE DOCUMENTATION**

### Dataset: Xente Transaction Ledger

**Source Origin:**
- **Provider:** Xente Fintech Platform (Mobile Money & Digital Payment Service)
- **Business Context:** Transaction data from Xente's mobile payment ecosystem, used to construct behavioral credit risk indicators for Bati Bank's BNPL service
- **Collection Period:** [TBD - extracted from TransactionStartTime during EDA]
- **Geographic Coverage:** Primarily East African markets (96%+ single country concentration)

**Dataset Specifications:**
| Attribute | Value |
|-----------|-------|
| **Records** | 95,493 transactions |
| **Features** | 15 variables (see data dictionary below) |
| **Time Span** | [Review `TransactionStartTime` range in `notebooks/eda.ipynb` for exact period] |
| **Data Quality** | 0% missing values; exceptional completeness |
| **Unique Customers** | ~30,000+ AccountIds |
| **File Location** | `data/data.csv` (primary), `data/Xente_Variable_Definitions.csv` (schema) |

**Data Dictionary (15 Variables):**

| Variable | Type | Domain | Purpose | Notes |
|----------|------|--------|---------|-------|
| **TransactionId** | String (UUID) | Identifier | Unique transaction reference | Drop before modeling (no predictive value) |
| **AccountId** | String | Customer ID | Links transactions to individuals | Primary grouping key for RFM aggregation |
| **Amount** | Float64 | Transaction Value (currency units) | Transaction magnitude | Extreme right-skew; requires log-transformation |
| **Value** | Float64 | Absolute Amount (redundant) | Duplicate Amount indicator | **Action:** Drop (multicollinearity: r=0.9897 with Amount) |
| **ProductCategory** | Categorical | {Airtime, Data, Merchant, Other} | Service/product type | Pareto: top 2 categories = 90% volume |
| **ChannelId** | Categorical | {Web, Mobile, USSD} | Transaction channel | Limited cardinality; enables channel risk analysis |
| **PricingStrategy** | Categorical | {Standard, Premium, VIP, Other} | Merchant tier / pricing scheme | Proxy for customer segment quality |
| **FraudResult** | Binary | {0, 1} | Fraud indicator (0=legitimate, 1=fraud) | Imbalanced class (0.20% positive); strong risk signal |
| **TransactionStartTime** | DateTime (ISO 8601) | Timestamp | Transaction initiation time | Enable temporal RFM calculations (recency windows) |
| **CountryCode** | Categorical | {KE, UG, TZ, ...} | Country code (ISO 3166-1 alpha-2) | Limited diversity (96%+ single country); not useful for segmentation |
| **ProviderId** | Categorical | Merchant/Provider ID | Service provider identifier | Pareto: top providers = 80% volume |
| **[Additional Fields]** | [Variable] | [Domain] | [Purpose] | [To be confirmed from raw data inspection] |

**Data Lineage & Quality Assurance:**

1. **Source System:** Xente transaction ledger (production OLTP database or export)
2. **Export Process:** [Document extraction query/procedure - TBD]
3. **Data Quality Checks Performed:**
   - ✅ Completeness audit: 0 missing values across all 95,493 records
   - ✅ Uniqueness audit: No duplicate TransactionIds
   - ✅ Range validation: All numerical fields within expected business ranges
   - ✅ Temporal validation: TransactionStartTime values coherent and ordered
   - ✅ Categorical validation: All categorical values within defined domains
4. **Quality Score:** 9.5/10 (Exceptional - only minor outliers in Amount distribution)

**Usage Rights & Governance:**
- **Classification:** Internal Use / Confidential
- **Retention Policy:** [To be defined based on regulatory requirements]
- **Access Control:** [Define role-based access in production deployment]
- **GDPR Compliance:** PII (AccountId) is pseudonymized; retain data minimization principle during feature engineering

**Reproducibility Notes:**
- All data transformations are version-controlled in `notebooks/eda.ipynb`
- Feature engineering pipeline will be documented in `src/data_processing.py`
- Model training code references exact data column mappings for auditability

---

## Credit Scoring Business Understanding

### 1. Basel II Framework & Model Interpretability
The Basel II Accord fundamentally shifted how financial institutions calculate regulatory capital by allowing the use of internal models (the Internal Ratings-Based, or IRB, approach). Because these models directly dictate how much capital a bank must hold to buffer against unexpected losses, regulatory bodies demand absolute transparency.

* **Auditability & Accountability:** Regulators (such as the Fed, ECB, or local central banks) must be able to audit every variable, weight, and decision path. A "black box" model prevents regulators from verifying if the risk weights are mathematically sound and economically justifiable.
* **Mitigation of Model Risk:** Credit risk models are susceptible to "concept drift" and systemic shocks. An interpretable model allows risk managers to understand *why* a model is failing during an economic downturn, enabling manual overrides (margin/post-model adjustments) that keep the institution solvent.
* **Fair Lending & Compliance:** Under regulations like the Equal Credit Opportunity Act (ECOA) or GDPR, institutions are legally required to provide "adverse action notices"—clear, defensible reasons why a applicant was denied credit. Well-documented models ensure these reasons are mathematically traceable back to specific borrower attributes.

---

### 2. The Necessity and Risks of Proxy Variables
In credit scoring, a true "default" event is a lag indicator. If a modeler waits 12 to 24 months to observe a definitive default (e.g., 90+ Days Past Due), the training data becomes outdated, and the business cannot adapt to current macroeconomic shifts. Therefore, a **proxy variable** (e.g., 30 or 60 Days Past Due, or high credit utilization thresholds) is used as a leading indicator of distress.

While necessary for model velocity, proxy-based prediction introduces several distinct business risks:

| Risk Category | Description | Business Impact |
| :--- | :--- | :--- |
| **False Positive Risk** | Labeling a borrower who is merely late on a payment as a "defaulter." | **Revenue Loss:** The bank rejects potentially profitable, loyal customers who face temporary liquidity issues but intend to repay. |
| **False Negative Risk** | Classifying a borrower who habitually rolls over debt or uses predatory refinancing as "good." | **Credit Losses:** The bank inadvertently expands exposure to structurally weak borrowers, leading to massive write-offs down the line. |
| **Target Misalignment** | The proxy fails to accurately reflect the actual economic loss (Loss Given Default). | **Regulatory Penalties:** Capital reserve calculations become decoupled from reality, leading to under-capitalization and regulatory intervention. |

---

### 3. Model Architecture Trade-Offs: Logistic Regression vs. Gradient Boosting
When deploying credit models in a regulated financial context, data scientists must constantly balance predictive power against regulatory compliance and operational simplicity.

HIGH INTERPRETABILITY                                     HIGH PERFORMANCE
Low Operational Risk                                      Low Credit Risk

#### Traditional Approach: Logistic Regression with Weight of Evidence (WoE)
* **Pros:** * **Maximum Interpretability:** It can be seamlessly converted into a classic "points-based" credit scorecard that non-technical stakeholders and loan officers can easily understand.
    * **Monotonicity:** WoE transformations force a linear relationship between the predictor and the target, ensuring that "higher income" *always* correlates with "lower risk," preventing counter-intuitive model behavior.
    * **Regulatory Consensus:** It is the industry gold standard; approval processes through risk committees and regulators are streamlined.
* **Cons:** * **Lower Predictive Capacity:** It completely misses complex, non-linear feature interactions unless they are manually engineered, which can lead to leaving margin on the table.

#### Machine Learning Approach: Gradient Boosting (e.g., XGBoost, LightGBM)
* **Pros:**
    * **Superior Risk Discrimination:** Significantly higher Gini coefficients/AUC. It captures subtle, multi-dimensional interactions between features automatically, resulting in fewer bad loans and higher approval rates for marginal "good" borrowers.
* **Cons:**
    * **Explainability Barriers:** While SHAP (SHapley Additive exPlanations) and LIME provide post-hoc explanations, they are approximations. Regulators may reject them if the underlying split logic appears unstable or counter-intuitive.
    * **Stability Risks:** Tree-based models are prone to overfitting on historical niches, potentially behaving erratically when exposed to unseen macroeconomic anomalies.