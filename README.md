# Loan Default Risk Analysis

This project analyzes personal loan defaults for Horizon Financial Group using borrower profiles and loan application data (2024–2025).

## Objective
Identify the key factors driving defaults and provide underwriting recommendations to reduce portfolio default rate.

## Dataset
- `loan_applications.csv`
- `borrower_profiles.csv`

Joined on `borrower_id`.

## Tools
- Python
- Pandas
- NumPy
- SciPy
- Matplotlib

## Workflow
1. Load and validate both datasets (types, null checks, summary stats).
2. Merge borrower and loan data on `borrower_id`.
3. Segment default rates by:
   - Credit score buckets
   - DTI ranges
   - Loan purpose
   - Employment status
   - Years employed (<2 vs 2+)
4. Run correlation analysis for numeric features against default outcome.
5. Run statistical tests:
   - Chi-square (employment tenure vs default)
   - Welch t-test and Mann-Whitney U (loan amount by default status)
6. Generate visualizations and summarize risk recommendations.

## Key Findings
- Overall default rate is about **24.3%**.
- Highest default rate occurs in the **520–599 credit score bucket**.
- Default risk increases as **DTI** rises, especially above **40%**.
- Borrowers with **<2 years employment** show significantly higher default risk.
- Some loan purposes (e.g., Wedding, Home Improvement) have higher default rates.
- Average loan amount is not significantly different between defaulted and non-defaulted groups.

## Recommendations
- Enforce stricter minimum credit score policy (e.g., prioritize 600+).
- Set DTI approval threshold around **40%** (or **35%** for stricter control).
- Add extra review for applicants with short employment tenure and high-risk loan purposes.

## Repository Contents
- `analyze_loan_defaults.py` — main analysis script
- `default_rate_credit_score.png`
- `dti_vs_default_scatter.png`
- `default_rate_loan_purpose.png`
