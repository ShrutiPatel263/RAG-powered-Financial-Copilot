import pandas as pd
from langchain_core.documents import Document

def load_csv(file_path: str) -> tuple:
    """
    Reads an expense CSV and returns:
    1. df         → the raw DataFrame (for analytics calculations)
    2. documents  → list of Document objects (for RAG)
    """
    
    # Read CSV into a DataFrame
    # A DataFrame is like an Excel table — rows and columns
    df = pd.read_csv(file_path)
    
    # --- NORMALIZE COLUMN NAMES ---
    # Users might have "Date", "DATE", "date" — make it uniform
    df.columns = df.columns.str.lower().str.strip()
    
    # Common column name variations → standardize them
    rename_map = {
        "transaction date": "date",
        "txn date":         "date",
        "description":      "merchant",
        "narration":        "merchant",
        "debit":            "amount",
        "withdrawal":       "amount",
        "spend":            "amount",
        "type":             "category",
        "category name":    "category",
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Parse date column so Python understands it as a date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Convert amount to number (remove commas like "1,200")
    if "amount" in df.columns:
        df["amount"] = (
            df["amount"]
            .astype(str)
            .str.replace(",", "")
            .pipe(pd.to_numeric, errors="coerce")
        )
    
    # If no category column, set default
    if "category" not in df.columns:
        df["category"] = "General"
    
    # --- CONVERT TO DOCUMENTS FOR RAG ---
    # Group transactions by month, make one Document per month
    documents = []
    
    if "date" in df.columns:
        df["month"] = df["date"].dt.to_period("M")
        
        for month, group in df.groupby("month"):
            # Build a text summary of this month's transactions
            text = f"Month: {month}\n"
            text += f"Number of transactions: {len(group)}\n"
            text += f"Total spent: Rs {group['amount'].sum():.0f}\n\n"
            text += "Transactions:\n"
            
            for _, row in group.iterrows():
                merchant = row.get("merchant", "Unknown")
                amount   = row.get("amount", 0)
                category = row.get("category", "General")
                text += f"  - {merchant}: Rs {amount:.0f} ({category})\n"
            
            # Wrap text in a Document object (same format as PDF)
            doc = Document(
                page_content=text,
                metadata={"source": file_path, "month": str(month), "type": "csv"}
            )
            documents.append(doc)
    
    return df, documents