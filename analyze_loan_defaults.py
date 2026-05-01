"""
Loan Default Risk Analysis — Horizon Financial Group
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(r"c:\Users\Sepid\Downloads")
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def main():
    loans = pd.read_csv(DATA_DIR / "loan_applications.csv")
    borrowers = pd.read_csv(DATA_DIR / "borrower_profiles.csv")

    print("=== 1. IMPORT & EXPLORE ===\n")
    print("loan_applications shape:", loans.shape)
    print(loans.dtypes)
    print("\nnull counts (loans):\n", loans.isnull().sum())
    print("\nborrower_profiles shape:", borrowers.shape)
    print(borrowers.dtypes)
    print("\nnull counts (borrowers):\n", borrowers.isnull().sum())
    print("\nBasic stats — loans (numeric):\n", loans.describe())
    print("\nBasic stats — borrowers (numeric):\n", borrowers.describe())

    df = loans.merge(borrowers, on="borrower_id", how="left", validate="m:1")
    join_miss = df["credit_score"].isna().sum()
    print(f"\nJoined rows: {len(df)}, unmatched borrower rows (missing profile): {join_miss}")

    overall_default = df["defaulted"].mean()
    print(f"\nOverall default rate: {overall_default:.1%} ({df['defaulted'].sum()} / {len(df)})")

    # Credit score buckets (spec: 520–599, 600–649, 650–699, 700–749, 750+)
    bins = [0, 519, 599, 649, 699, 749, float("inf")]
    labels = ["<520", "520-599", "600-649", "650-699", "700-749", "750+"]
    df["credit_bucket"] = pd.cut(
        df["credit_score"], bins=bins, labels=labels, right=True
    )

    print("\n=== DEFAULT RATE BY CREDIT SCORE BUCKET ===")
    cs = (
        df.groupby("credit_bucket", observed=True)
        .agg(loans=("defaulted", "size"), defaults=("defaulted", "sum"))
        .assign(default_rate=lambda x: x["defaults"] / x["loans"])
        .sort_values("default_rate", ascending=False)
    )
    print(cs.to_string())

    # DTI bins for segment view
    dti_bins = [0, 30, 40, 50, 60, 100]
    dti_labels = ["0-30%", "31-40%", "41-50%", "51-60%", "61%+"]
    df["dti_bin"] = pd.cut(df["dti_ratio"], bins=dti_bins, labels=dti_labels, right=True)

    print("\n=== DEFAULT RATE BY DTI BIN ===")
    print(
        df.groupby("dti_bin", observed=True)
        .agg(loans=("defaulted", "size"), defaults=("defaulted", "sum"))
        .assign(default_rate=lambda x: x["defaults"] / x["loans"])
        .to_string()
    )

    print("\n=== DEFAULT RATE BY LOAN PURPOSE ===")
    purpose = (
        df.groupby("loan_purpose")
        .agg(loans=("defaulted", "size"), defaults=("defaulted", "sum"))
        .assign(default_rate=lambda x: x["defaults"] / x["loans"])
        .sort_values("default_rate", ascending=False)
    )
    print(purpose.to_string())

    print("\n=== DEFAULT RATE BY EMPLOYMENT STATUS ===")
    print(
        df.groupby("employment_status")
        .agg(loans=("defaulted", "size"), defaults=("defaulted", "sum"))
        .assign(default_rate=lambda x: x["defaults"] / x["loans"])
        .sort_values("default_rate", ascending=False)
        .to_string()
    )

    # Years employed < 2 vs >= 2
    df["emp_lt2"] = df["years_employed"] < 2
    lt2 = df.groupby("emp_lt2")["defaulted"].agg(["sum", "count", "mean"])
    lt2.index = ["2+ years employed", "Under 2 years employed"]
    print("\n=== YEARS EMPLOYED (under 2 vs 2+ years) ===")
    print(lt2.to_string())

    # Chi-square: employment tenure bucket vs default
    ct = pd.crosstab(df["emp_lt2"], df["defaulted"])
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
    print(f"\nChi-square (tenure <2 vs default): chi2={chi2:.3f}, p={p_chi:.4g}")

    # Loan amount: defaulted vs not
    def_amt = df.loc[df["defaulted"] == 1, "loan_amount"]
    ok_amt = df.loc[df["defaulted"] == 0, "loan_amount"]
    t_stat, p_t = stats.ttest_ind(def_amt, ok_amt, equal_var=False)
    mw_stat, p_mw = stats.mannwhitneyu(def_amt, ok_amt, alternative="two-sided")
    print("\n=== LOAN AMOUNT: DEFAULT vs NON-DEFAULT ===")
    print(f"Defaulted mean: ${def_amt.mean():,.2f}, median: ${def_amt.median():,.2f}")
    print(f"Non-default mean: ${ok_amt.mean():,.2f}, median: ${ok_amt.median():,.2f}")
    print(f"Welch t-test p-value: {p_t:.4g}")
    print(f"Mann-Whitney U p-value: {p_mw:.4g}")

    # Correlations with defaulted (numeric)
    num_cols = [
        "credit_score",
        "dti_ratio",
        "annual_income",
        "loan_amount",
        "interest_rate",
        "monthly_payment",
        "term_months",
        "age",
        "years_employed",
        "dependents",
        "existing_monthly_debt",
    ]
    corr_with_default = df[num_cols + ["defaulted"]].corr(numeric_only=True)["defaulted"].drop("defaulted").sort_values(key=abs, ascending=False)
    print("\n=== CORRELATION WITH DEFAULTED (Pearson) ===")
    print(corr_with_default.to_string())

    # DTI threshold sweep: suggest cutoff where default rate jumps or cumulative lift
    print("\n=== DTI THRESHOLD SWEEP (approve if DTI <= threshold) ===")
    thresholds = list(range(25, 81, 5))
    rows = []
    for t in thresholds:
        sub = df[df["dti_ratio"] <= t]
        if len(sub) < 10:
            continue
        dr = sub["defaulted"].mean()
        share = len(sub) / len(df)
        rows.append({"max_dti": t, "n_loans": len(sub), "pct_book": share, "default_rate": dr})
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False))

    # --- Charts ---
    fig, ax = plt.subplots(figsize=(8, 5))
    cs_plot = (
        df.groupby("credit_bucket", observed=True)["defaulted"]
        .mean()
        .reindex(labels)
    )
    cs_plot.plot(kind="bar", color="#c0392b", ax=ax)
    ax.set_ylabel("Default rate")
    ax.set_xlabel("Credit score range")
    ax.set_title("Default rate by credit score bucket")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "default_rate_credit_score.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        df.loc[df["defaulted"] == 0, "dti_ratio"],
        df.loc[df["defaulted"] == 0, "defaulted"],
        alpha=0.35,
        label="No default",
        s=22,
        c="#2980b9",
    )
    ax.scatter(
        df.loc[df["defaulted"] == 1, "dti_ratio"],
        df.loc[df["defaulted"] == 1, "defaulted"],
        alpha=0.6,
        label="Default",
        s=28,
        c="#c0392b",
    )
    ax.set_xlabel("DTI ratio (%)")
    ax.set_ylabel("Defaulted (0/1)")
    ax.set_title("DTI vs default status")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dti_vs_default_scatter.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    purpose_sorted = purpose.sort_values("default_rate", ascending=True)
    ax.barh(purpose_sorted.index, purpose_sorted["default_rate"], color="#8e44ad")
    ax.set_xlabel("Default rate")
    ax.set_title("Default rate by loan purpose")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "default_rate_loan_purpose.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
