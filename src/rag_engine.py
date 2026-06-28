import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document

from src.knowledge import get_finance_knowledge

load_dotenv()  # reads your .env file

# Step 1 : Split the documents into chunks
def split_documents(documents: list) -> list:
    """
    Takes Documents (from PDF or CSV loader) and splits them
    into smaller pieces called "chunks".
    
    Why split? A whole bank statement is too long to embed as one piece.
    Smaller chunks = more precise retrieval.
    
    RecursiveCharacterTextSplitter splits by paragraph first,
    then sentence, then word — always tries to keep meaning intact.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk = max 500 characters
        chunk_overlap=50,    # 50 chars overlap so context isn't lost at edges
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


# Step 2: Built the vector store
def build_vector_store(user_documents: list) -> Chroma:
    """
    1. Takes user Documents (from PDF + CSV)
    2. Adds finance knowledge Documents
    3. Splits all into chunks
    4. Embeds each chunk (converts text → numbers)
    5. Stores in ChromaDB on disk
    
    Returns the ChromaDB collection so we can query it later.
    """
    
    # Get finance knowledge docs
    knowledge_docs = get_finance_knowledge()
    
    # Combine user data + knowledge into one list
    all_documents = user_documents + knowledge_docs
    
    # Split into chunks
    chunks = split_documents(all_documents)
    
    # Create embeddings model
    # Embeddings = converts text into a list of numbers (vector)
    # Similar texts get similar number patterns
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Store in ChromaDB
    # persist_directory = folder where ChromaDB saves data to disk
    # So your data survives between app restarts
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Vector store built and saved to ./chroma_db")
    return vector_store


# Step 3: Query the RAG System
def ask_question(question: str, vector_store: Chroma, analytics_text: str) -> str:
    """
    The full RAG query pipeline:
    1. Embed the user's question
    2. Find the 5 most similar chunks in ChromaDB
    3. Build a prompt with retrieved context + analytics
    4. Send to Gemini and return the answer
    """
    
    # Retrieve top 5 most relevant chunks
    # similarity_search embeds your question and finds closest chunks
    retrieved_docs = vector_store.similarity_search(question, k=5)
    
    # Combine retrieved chunk texts into one context string
    context = "\n\n---\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )
    
    # Build the prompt
    # inject BOTH the hard analytics AND the retrieved context
    prompt = f"""You are a personal financial advisor.
Answer the user's question using ONLY the data provided below.
Be specific: mention exact amounts and categories.
If the user is overspending somewhere, say so clearly and give a fix.

{analytics_text}

RETRIEVED FINANCIAL CONTEXT:
{context}

User question: {question}

Answer:"""
    
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    response = llm.invoke(prompt)

# Gemini sometimes returns content as a list
    if isinstance(response.content, list):

        text = ""

        for block in response.content:


            if isinstance(block, dict) and "text" in block:

                text += block["text"]

            return text

    return str(response.content)
