import streamlit as st
import os
import pandas as pd
import tempfile
import import_ipynb

# Import all our custom modules

from src.pdf_loader import load_multiple_pdfs
from src.csv_loader import load_csv
from src.analytics import compute_analytics, analytics_to_text
from src.rag_engine import build_vector_store, ask_question

# --- PAGE CONFIG ---
# Must be the first Streamlit command
st.set_page_config(
    page_title="Financial Advisor AI",
    page_icon="💰",
    layout="wide"   # uses full screen width
)

st.title("AI Financial Advisor")
st.caption("Upload your bank statements and expense files to get personalized insights")

# --- SESSION STATE ---
# Streamlit reruns your whole script on every interaction.
# st.session_state stores data that should PERSIST between reruns.
# Think of it as a memory box for the app.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None   # ChromaDB collection
if "analytics" not in st.session_state:
    st.session_state.analytics = None      # computed numbers
if "analytics_text" not in st.session_state:
    st.session_state.analytics_text = ""  # text version for LLM
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of {role, content} dicts
if "df" not in st.session_state:
    st.session_state.df = None            # the pandas DataFrame

# --- LAYOUT: TWO COLUMNS ---
# col1 = left sidebar for uploads, col2 = right area for chat
col1, col2 = st.columns([1, 2])  # col2 is twice as wide

# =====================
# LEFT COLUMN: UPLOAD
# =====================
with col1:
    st.subheader("Upload Documents")
    
    # PDF uploader — accepts multiple files
    pdf_files = st.file_uploader(
        "Bank / Credit Card Statements (PDF)",
        type=["pdf"],
        accept_multiple_files=True,  # user can upload several PDFs
        help="Upload your bank statement PDFs"
    )
    
    # CSV uploader
    csv_files = st.file_uploader(
        "Expense CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload CSVs with columns: Date, Category, Amount"
    )
    
    # Process button
    if st.button("Analyze my finances", type="primary"):
        
        if not pdf_files and not csv_files:
            st.warning("Please upload at least one file first.")
        else:
            # Show a spinner while processing
            with st.spinner("Reading and processing your documents..."):
                
                all_documents = []  # will hold all LangChain Documents
                all_dfs = []        # will hold all DataFrames
                
                # --- PROCESS PDFs ---
                if pdf_files:
                    # Streamlit gives us file objects, not file paths.
                    # PyPDFLoader needs a file PATH.
                    # Solution: save to a temp file, get its path.
                    temp_pdf_paths = []
                    for pdf_file in pdf_files:
                        # NamedTemporaryFile creates a temp file on disk
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(pdf_file.read())  # save to disk
                            temp_pdf_paths.append(tmp.name)
                    
                    pdf_docs = load_multiple_pdfs(temp_pdf_paths)
                    all_documents.extend(pdf_docs)
                    
                    # Clean up temp files
                    for path in temp_pdf_paths:
                        os.unlink(path)
                
                # --- PROCESS CSVs ---
                if csv_files:
                    for csv_file in csv_files:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".csv"
                        ) as tmp:
                            tmp.write(csv_file.read())
                            tmp_path = tmp.name
                        
                        df, csv_docs = load_csv(tmp_path)
                        all_documents.extend(csv_docs)
                        all_dfs.append(df)
                        os.unlink(tmp_path)
                    
                    # Combine all CSVs into one DataFrame
                    if all_dfs:
                        combined_df = pd.concat(all_dfs, ignore_index=True)
                        st.session_state.df = combined_df
                        
                        # Run analytics
                        analytics = compute_analytics(combined_df)
                        st.session_state.analytics = analytics
                        st.session_state.analytics_text = analytics_to_text(analytics)
                
                # --- BUILD RAG ---
                with st.spinner("Building AI knowledge base (this takes ~30 seconds)..."):
                    vector_store = build_vector_store(all_documents)
                    st.session_state.vector_store = vector_store
                
                st.success("Done! Ask me anything about your finances below.")
    
    # --- SHOW ANALYTICS DASHBOARD ---
    if st.session_state.analytics:
        st.divider()
        st.subheader("Your Financial Snapshot")
        a = st.session_state.analytics
        
        # Metric cards — Streamlit's built-in stat display
        st.metric("Total Spent", f"Rs {a['total_spent']:,.0f}")
        st.metric("Avg per Month", f"Rs {a['avg_monthly']:,.0f}")
        st.metric("Health Score", f"{a['health_score']}/100")
        
        # Bar chart of spending by category
        if a["by_category"]:
            st.subheader("Spending by Category")
            cat_df = pd.DataFrame(
                list(a["by_category"].items()),
                columns=["Category", "Amount"]
            )
            st.bar_chart(cat_df.set_index("Category"))

# =====================
# RIGHT COLUMN: CHAT
# =====================
with col2:
    st.subheader("Chat with your Financial Advisor")
    
    # Show placeholder if not ready
    if not st.session_state.vector_store:
        st.info("Upload your financial documents and click 'Analyze my finances' to start chatting.")
    
    else:
        # --- DISPLAY CHAT HISTORY ---
        # Loop through saved messages and display each one
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):  # "user" or "assistant"
                st.markdown(message["content"])
        
        # --- CHAT INPUT BOX ---
        # st.chat_input shows a text box at the bottom
        # It returns None until user presses Enter
        user_question = st.chat_input("Ask anything: 'Where am I overspending?' or 'How can I save more?'")
        
        if user_question:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Save user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Get answer from RAG
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(
                        question=user_question,
                        vector_store=st.session_state.vector_store,
                        analytics_text=st.session_state.analytics_text
                    )
                st.markdown(answer)
            
            # Save assistant reply to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
        
        # Quick question buttons
        st.divider()
        st.caption("Quick questions:")
        q_cols = st.columns(2)
        questions = [
            "Where am I overspending?",
            "How can I increase my savings?",
            "Which category should I cut first?",
            "Am I following the 50/30/20 rule?"
        ]
        for i, q in enumerate(questions):
            with q_cols[i % 2]:
                if st.button(q, key=f"q_{i}"):
                    # Clicking a button triggers a rerun
                    # We set a flag in session_state to trigger the question
                    st.session_state.chat_history.append({"role":"user","content":q})
                    answer = ask_question(q, st.session_state.vector_store, st.session_state.analytics_text)
                    st.session_state.chat_history.append({"role":"assistant","content":answer})
                    st.rerun()