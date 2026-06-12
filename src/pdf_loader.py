import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str) -> list:
    """
    Reads a PDF file and returns a list of Document objects.
    
    What is a Document?
    - doc.page_content  → the actual text from that page
    - doc.metadata      → {'source': 'file.pdf', 'page': 0}
    
    PyPDFLoader gives you one Document PER PAGE.
    So a 5-page bank statement = 5 Documents.
    """
    
    # PyPDFLoader needs a file path (not the file itself)
    loader = PyPDFLoader(file_path)
    
    # .load() reads all pages and returns a list
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages from {file_path}")
    
    return documents


def load_multiple_pdfs(file_paths: list) -> list:
    """
    Loop through multiple PDF files and combine all pages.
    Returns one big list of all Documents from all PDFs.
    """
    all_documents = []
    
    for path in file_paths:
        docs = load_pdf(path)
        all_documents.extend(docs)  # add to the main list
    
    return all_documents