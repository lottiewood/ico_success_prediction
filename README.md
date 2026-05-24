# ICO Fundraising Success Prediction

## Overview
This project develops machine learning models to predict whether Initial Coin 
Offering (ICO) fundraising campaigns will successfully reach their funding goals. 
Using a dataset of 5,828 real-world ICO campaigns sourced from IcoBench, the 
analysis covers the full data science pipeline: from raw data cleaning through 
to model evaluation.

---

## Business Context
ICOs are a blockchain-based fundraising mechanism in which companies raise capital 
by selling digital tokens to investors. This structure creates a high-stakes environment for both entrepreneurs and investors, underscoring the importance of understanding why ICOs succeed or fail, and the ability to predict a project’s outcome.

---

## Analytical Pipeline 

The script is structured into four clearly commented sections:

### 1. Data Understanding
- Summary statistics and missing value analysis
- Distribution visualisations for continuous and binary variables
- Class imbalance assessment (34% success / 66% failure)
- Identification of data quality issues (inconsistent encoding, outliers, 
  impossible values)

### 2. Data Preparation
**2.1 Handling Incorrect Values and Outliers**
- Standardisation of inconsistently encoded binary variables (`whitelist`, `kyc`, 
  `bonus`, `mvp`, `ERC20`) using a custom `binarise()` function
- Date parsing using `lubridate`
- Removal of rows with impossible values (negative durations, end dates before 
  start dates)
- Manual cross-referencing confirmed duration errors, durations >365 days were considered outliers and likely data entry mistakes and were deleted
- Removal of `price_usd` outliers above the 99th percentile
- Correction of `distributed_in_ico` scale inconsistency (proportion vs percentage)

**2.2 Feature Engineering**
- `duration`: campaign length in days (derived from ICO start/end dates)
- `pre_duration`: presale length in days
- `had_pre_ico`: binary indicator for presale presence
- `has_whitepaper`, `has_linkedin`, `has_github`, `has_website`: binary URL 
  presence flags (transparency signals)
- `is_US_Sing_UK`: binary indicator for top-3 ICO countries
- `log_token_for_sale`: log-transformed token supply (log-normal distribution)
- `btc_price`: Bitcoin closing price at ICO launch date, sourced from Yahoo 
  Finance via `quantmod` as a proxy for cryptocurrency market sentiment

**2.3 Missing Value Handling**
- Columns with >70% missingness dropped (`sold_tokens`, `pre_ico_price_usd`)
- Listwise deletion for `duration`, `btc_price`, `price_usd` - skewed 
  distributions make imputation unreliable and lower proportions of missingness mean data loss is not significant
- Median imputation for `rating`, `teamsize`, `distributed_in_ico`, 
  `log_token_for_sale` as higher proportions of missingess and missingness consistent with MCAR
- Before/after distribution comparison to verify imputation did not introduce bias

**2.4 Exploratory Relationship Analysis**
- Correlation matrix of continuous predictors (max |r| = 0.33, no multicollinearity)
- Boxplots of continuous predictors by ICO outcome - `rating`, `teamsize` and `duration exhibit strongest association with `goal`
- Stacked barplots of binary predictors by ICO outcome - URL indicator variables show relationship with `goal` indicating transparency may be significant in determining the outcome of an ICO

Final cleaned dataset: **4,475 rows × 19 predictor variables**

### 3. Modelling
All models tuned using **10-fold cross-validation** on the training set with 
**AUC-ROC** as the selection criterion, chosen for robustness to class imbalance. 
Final evaluation on a held-out test set (80/20 split, stratified by outcome).

| Model | Algorithm | Key Hyperparameter Tuned |
|---|---|---|
| Decision Tree | C5.0 with adaptive boosting | Number of boosting trials (1–20) |
| Random Forest | `randomForest` | Number of trees (50–500) |
| SVM | `ksvm` | Cost C (0.01–10) × kernel (linear, RBF, polynomial) |
| KNN | `knn` | Number of neighbours K (1–121) |

### 4. Evaluation
- Accuracy, Precision, Recall, F1 score on held-out test set
- ROC curves and AUC scores for all four models

---

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.676 | 0.638 | 0.439 | 0.520 | 0.724 |
| Random Forest | 0.685 | 0.660 | 0.439 | 0.527 | 0.733 |
| SVM | 0.726 | 0.694 | 0.536 | 0.605 | 0.721 |
| KNN | 0.675 | 0.645 | 0.416 | 0.506 | 0.719 |

**Random Forest** achieved the highest AUC (0.733), making it preferable for 
ranking ICO campaigns by success probability. **SVM** achieved the strongest 
classification metrics at the default 0.5 threshold (F1 = 0.605), making it 
preferable for binary go/no-go decisions.

**Strongest predictors** (Random Forest variable importance): expert `rating`, 
`btc_price` at launch, campaign `duration`, and `teamsize`.


