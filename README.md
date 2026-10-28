# Loan Approval Classification

Classification project : predict whether a loan application will be approved.

**Live demo:** _add your Streamlit link here_

## Problem
Given an applicant's education, income, work experience, home ownership, loan amount, loan purpose, interest rate, credit score and previous-default status, predict whether the loan is approved (`loan_status`: 1 = approved, 0 = rejected).

## Dataset
- Loan Approval Classification Dataset (Kaggle, by Ta-wei Lo), file `loan_data.csv`
- 45,000 rows, 14 columns. **Synthetic data** (generated with SMOTENC), so results show the method, not real bank behaviour.
- Target: 77.78% rejected, 22.22% approved (imbalanced).
- The dataset has no employment-status column, so work experience and home ownership are used instead.

## Approach
1. **Explore:** no missing values, no duplicates, impossible values found (age up to 144, experience up to 125 years, income up to 7.2 million).
2. **Clean:** removed 28 rows with impossible or extreme values (45,000 to 44,972), encoded categories (0/1, ordered education, one-hot for home ownership and loan purpose).
3. **EDA:** 12 plots, approval rates by gender, income, experience, education, home ownership, loan purpose, previous default, credit score, loan-to-income ratio, boxplots, correlation matrix.
4. **Features:** dropped `person_gender` (no effect, sensitive attribute), `person_age` and `cb_person_cred_hist_length` (redundant with experience). 16 features remain.
5. **Imbalance:** stratified 80/20 split, `class_weight="balanced"`, plus a SMOTENC experiment on the training data only.
6. **Models:** Logistic Regression, Decision Tree, Random Forest, SVM (trained on a 15,000-row sample for speed), tuned Random Forest (RandomizedSearchCV).
7. **Evaluate:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves, threshold analysis, permutation importance.

## Results (test set, 8,995 applications)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (tuned)** | 0.926 | 0.826 | 0.842 | **0.834** | 0.974 |
| Random Forest | 0.928 | 0.896 | 0.764 | 0.824 | 0.975 |
| Random Forest + SMOTENC | 0.914 | 0.782 | 0.853 | 0.816 | 0.971 |
| Decision Tree (no limit) | 0.899 | 0.775 | 0.772 | 0.773 | 0.854 |
| SVM (15k sample) | 0.866 | 0.637 | 0.923 | 0.754 | 0.959 |
| Logistic Regression + SMOTENC | 0.871 | 0.662 | 0.863 | 0.749 | 0.949 |
| Logistic Regression | 0.853 | 0.613 | 0.918 | 0.735 | 0.954 |
| Decision Tree (max depth 6) | 0.845 | 0.595 | 0.941 | 0.729 | 0.960 |

Best settings: `n_estimators=100, max_depth=20, min_samples_split=10, min_samples_leaf=1, max_features=0.5, class_weight="balanced"`.

## Key findings
- Every applicant with a previous default is rejected (0% approval), so accuracy overstates model quality. Among no-default applicants, always predicting "rejected" scores only 0.551 accuracy.
- Top features: previous default, loan-to-income ratio, interest rate, income, renting.
- SMOTENC did not clearly beat class weights.
- Threshold choice trades wrong approvals against missed good customers (see the notebook).

## Limitations
- Synthetic data. Credit score has almost no effect, and approved applications tend to have higher loan-to-income ratios, higher rates and lower income, which is the opposite of normal lending practice.
- `loan_int_rate` is normally set after a decision, so it may leak information about the outcome.
- Not for real lending decisions. Validate on real data first.

## How to run
1. Open `notebook/Loan_Approval_Classification.ipynb` in Google Colab.
2. Upload `loan_data.csv` when asked.
3. Run **Runtime > Run all**.

### Run the web app
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
`app.py` needs `loan_model.pkl` and `loan_features.json` in the same folder (created by the last notebook cell).

## Project structure
```
data/        loan_data.csv
notebook/    Loan_Approval_Classification.ipynb
report/      Project report (PDF or Word)
app.py       Streamlit demo
requirements.txt
loan_model.pkl
loan_features.json
README.md
```

## Tech
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib, Streamlit.
