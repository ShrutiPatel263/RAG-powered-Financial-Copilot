
# 💰 RAG-powered Financial Copilot

> A document intelligence system that analyzes your bank statements and expense data using Retrieval-Augmented Generation (RAG) to deliver personalized financial insights through a natural language chat interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat&logo=streamlit)
![Gemini](https://img.shields.io/badge/Gemini-3%20Flash-blue?style=flat&logo=google)

---

## 📌 What is this project?

Most people don't know where their money goes. This app lets you upload your bank statements (PDF) and expense files (CSV), and ask questions in plain English like:

- *"Where am I overspending?"*
- *"Which category should I cut first?"*
- *"Am I following the 50/30/20 rule?"*
- *"How much did I spend on Food in January?"*

The system doesn't just retrieve text — it computes real analytics first, then uses RAG to combine your personal financial data with finance domain knowledge to give grounded, specific answers.

---

## 🏗️ Architecture

```
PDF / CSV Upload
      ↓
Document Parsing
(PyPDFLoader + pandas)
      ↓
Analytics Engine          ← computes totals, savings rate, health score
      ↓
Chunking + Embedding      ← RecursiveCharacterTextSplitter + Gemini Embeddings
      ↓
ChromaDB Vector Store     ← stores User Data + Finance Knowledge together
      ↓
Similarity Search         ← retrieves top-k relevant chunks
      ↓
Gemini 2.5 Flash          ← generates grounded financial advice
      ↓
Streamlit Chat Interface
```

### Why RAG over fine-tuning?

User financial data is personal and changes every month. Fine-tuning is for static domain knowledge. RAG lets us inject fresh user data at query time without retraining — making it the right choice for this use case.

---

## ✨ Features

- 📄 **Multi-format ingestion** — PDF bank statements via PyPDFLoader, expense CSVs via pandas
- 📊 **Pre-RAG analytics engine** — computes total spend, monthly averages, category breakdown, and financial health score before any LLM call
- 🧠 **Dual-knowledge vector store** — user transaction data and finance domain knowledge (50/30/20 rule, emergency fund, debt strategies) stored together in ChromaDB
- 💬 **Natural language chat** — ask anything about your finances in plain English
- 📈 **Live dashboard** — spending by category bar chart, health score, monthly averages
- ⚡ **Quick question buttons** — one-click access to common financial queries
- 🔒 **Local storage** — all data stays on your machine, nothing sent to external servers

---

## 🗂️ Project Structure

```
financial_copilot/
│
├── app.py                  ← Streamlit UI (run this file)
├── requirements.txt        ← all dependencies
├── .env                    ← API keys (never commit this)
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py         ← makes src a Python package (empty file)
│   ├── pdf_loader.py       ← reads PDF bank statements using PyPDFLoader
│   ├── csv_loader.py       ← reads expense CSVs, normalizes columns, creates chunks
│   ├── analytics.py        ← computes spending totals, health score, category breakdown
│   ├── rag_engine.py       ← embeds chunks, builds ChromaDB, handles similarity search + LLM call
│   └── knowledge.py        ← hardcoded finance domain knowledge documents
│
├── data/
│   └── uploads/            ← temporary storage for uploaded files
│
└── chroma_db/              ← ChromaDB saves vector store here automatically
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | LangChain PyPDFLoader |
| CSV Processing | pandas |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | Google Gemini `embedding-001` |
| Vector Store | ChromaDB (local, persistent) |
| LLM | gemini-3-flash-preview |
| Framework | LangChain |
| Environment | python-dotenv |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/financial-copilot.git
cd financial-copilot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📋 Requirements

```
streamlit
langchain
langchain-community
langchain-google-genai
chromadb
pandas
python-dotenv
pypdf
```

---

## 🧪 How to Test

### Sample CSV format

```csv
Date,Category,Amount,Merchant
2026-01-05,Food,450,Swiggy
2026-01-08,Travel,1200,Uber
2026-01-12,Shopping,3500,Amazon
2026-01-15,Food,320,Zomato
2026-01-18,Utilities,800,Electricity Board
```

### Sample questions to ask

**Spending analysis**
- `How much did I spend in total?`
- `What is my biggest spending category?`
- `How much did I spend on Food?`
- `Compare my spending in January vs February`

**Finance knowledge**
- `Am I following the 50/30/20 rule?`
- `How much should my emergency fund be?`
- `Which category should I cut first to save more?`

**Combined insights**
- `Why am I not saving enough?`
- `Give me a savings plan based on my spending`
- `What does my financial health score mean?`

---

## 🔑 Key RAG Design Decisions

**Chunking by time period, not token count**

Transactions are grouped by month rather than split at arbitrary token limits. A chunk like *"January 2026 — Food — 8 transactions — Rs 5,000 total"* is semantically meaningful and retrieves correctly when a user asks about January spending.

**Analytics before RAG**

Hard numbers (totals, savings rate, health score) are computed from pandas before any embedding happens. These are injected directly into the LLM prompt as ground truth — the LLM never has to infer your total spend from retrieved text.

**Dual-knowledge vector store**

Both user transaction chunks and finance domain knowledge (budgeting rules, debt strategies, savings guidelines) are stored in the same ChromaDB collection. A single similarity search retrieves both personal data and relevant advice together, which is what makes the answers specific and actionable rather than generic.


## 🙋 Author

**Shruti Patel**
- GitHub: [@ShrutiPatel263](https://github.com/ShrutiPatel263)
- LinkedIn: [ShrutiPatel26](https://linkedin.com/in/ShrutiPatel26)

---
