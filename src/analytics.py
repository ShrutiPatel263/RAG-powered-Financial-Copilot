import pandas as pd

def compute_analytics(df: pd.DataFrame) -> dict:
    """
    Takes the DataFrame from csv_loader and computes
    all the key financial numbers.
    
    Returns a dictionary with all computed values.
    A dictionary is like a labeled box:
    analytics["total_spent"] gives you the total.
    """
    
    analytics = {}
    
    # --- TOTAL SPENT ---
    # Only count positive amounts (expenses, not credits/income)
    expenses = df[df["amount"] > 0]
    analytics["total_spent"] = expenses["amount"].sum()
    
    # --- SPENDING BY CATEGORY ---
    # groupby: split DataFrame into groups by category
    # sum: add up amounts in each group
    # sort: biggest category first
    if "category" in df.columns:
        cat_spending = (
            expenses
            .groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
        )
        analytics["by_category"] = cat_spending.to_dict()
        analytics["top_category"] = cat_spending.index[0] if len(cat_spending) > 0 else "N/A"
    else:
        analytics["by_category"] = {}
        analytics["top_category"] = "N/A"
    
    # --- MONTHLY BREAKDOWN ---
    if "date" in df.columns:
        df["month"] = df["date"].dt.to_period("M")
        monthly = expenses.groupby("month")["amount"].sum()
        analytics["by_month"] = {str(k): v for k, v in monthly.items()}
        analytics["num_months"] = len(monthly)
    else:
        analytics["by_month"] = {}
        analytics["num_months"] = 1
    
    # --- AVERAGE MONTHLY SPEND ---
    n = analytics["num_months"] if analytics["num_months"] > 0 else 1
    analytics["avg_monthly"] = analytics["total_spent"] / n
    
    # --- FINANCIAL HEALTH SCORE (0-100) ---
    score = 100
    
    # Check if any category is > 40% of total spend (overspending signal)
    for cat, amount in analytics["by_category"].items():
        pct = (amount / analytics["total_spent"] * 100) if analytics["total_spent"] > 0 else 0
        if pct > 40:
            score -= 20  # deduct for dominant category
    
    analytics["health_score"] = max(0, min(100, score))
    
    return analytics


def analytics_to_text(analytics: dict) -> str:
    """
    Convert analytics dict to a readable text string.
    This text gets injected into the LLM prompt as hard facts.
    """
    lines = [
        "=== USER FINANCIAL SUMMARY ===",
        f"Total spent: Rs {analytics['total_spent']:.0f}",
        f"Average per month: Rs {analytics['avg_monthly']:.0f}",
        f"Financial health score: {analytics['health_score']}/100",
        f"Biggest spending category: {analytics['top_category']}",
        "",
        "Spending by category:",
    ]
    
    for cat, amt in analytics["by_category"].items():
        pct = (amt / analytics["total_spent"] * 100) if analytics["total_spent"] > 0 else 0
        lines.append(f"  {cat}: Rs {amt:.0f} ({pct:.1f}%)")
    
    return "\n".join(lines)