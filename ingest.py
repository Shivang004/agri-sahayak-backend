#!/usr/bin/env python3
"""
A complete document ingestion pipeline using Gemini 2.5 Pro for OCR.

This script defines a DocumentIngester class that:
1.  Performs OCR on PDFs using the Gemini 2.5 Pro multimodal model.
2.  Parses the structured OCR output.
3.  Stores the page content and metadata in an SQLite database.
4.  Builds and saves a dense FAISS index using Gemini embeddings.
5.  Builds and saves a sparse BM25 index for keyword search.
"""

import os
import re
import sqlite3
import pickle
import faiss
import numpy as np
import google.generativeai as genai
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple

# Load environment variables from .env file
load_dotenv()

class DocumentIngester:
    """
    Handles the end-to-end process of PDF ingestion and indexing.
    """
    def __init__(self, data_dir="./data", db_path="db/documents.sqlite", index_dir="index"):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.index_dir = Path(index_dir)

        # Create necessary directories
        self.data_dir.mkdir(exist_ok=True)
        self.db_path.parent.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        self.faiss_index_path = self.index_dir / "faiss_index.bin"
        self.doc_map_path = self.index_dir / "doc_map.pkl"
        self.bm25_index_path = self.index_dir / "bm25_index.pkl"

        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDJDr6b_t_UF7nHd3eDk0cOvknL7O4ot9g")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        
        self.ocr_model = genai.GenerativeModel('models/gemini-2.5-pro')
        self.embedding_model_name = 'models/embedding-001'

        self.conn = None
        self.cursor = None

    def __enter__(self):
        """Connect to the database and initialize the table."""
        print("[DB] Connecting to database...")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._init_db()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Commit changes and close the database connection."""
        if self.conn:
            if exc_type is None:
                print("[DB] Committing all changes...")
                self.conn.commit()
            else:
                print(f"[DB] An error occurred: {exc_val}. Rolling back changes.")
                self.conn.rollback()
            self.conn.close()
            print("[DB] Database connection closed.")

    def _init_db(self):
        """Creates the documents table if it doesn't already exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(filename, page_number)
            )
        """)
        print("[DB] Database initialized successfully. 📚")

    def _ocr_pdf_with_gemini(self, pdf_path: Path) -> str | None:
        """Uploads a PDF and uses Gemini 2.5 Pro to perform OCR."""
        print(f"  [OCR] Starting OCR for '{pdf_path.name}'...")
        try:
            pdf_file = genai.upload_file(path=pdf_path, display_name=pdf_path.name)
            print(f"    - File uploaded as: {pdf_file.display_name}")
        except Exception as e:
            print(f"    ❌ Error uploading file: {e}")
            return None

        prompt = """
        Perform a complete and accurate OCR on the provided PDF file, page by page.
        Your output MUST strictly follow this format:
        1. Start with the text of the first page.
        2. After page N, insert the separator: '==End of OCR for page N=='
        3. Immediately after, insert the separator: '==Start of OCR for page N+1=='
        
        Example:
        ...text from page 1...
        ==End of OCR for page 1==
        ==Start of OCR for page 2==
        ...text from page 2...
        """
        
        print("    - Sending request to Gemini 2.5 Pro. This may take some time...")
        try:
            response = self.ocr_model.generate_content([prompt, pdf_file], request_options={"timeout": 600})
            print("    - Received OCR response from model.")
            return response.text
        except Exception as e:
            print(f"    ❌ An error occurred during model generation: {e}")
            return None

    def _parse_gemini_ocr_output(self, ocr_text: str, filename: str) -> List[Dict[str, Any]]:
        """Parses the raw OCR text from Gemini into structured page chunks."""
        documents = []
        separator_pattern = re.compile(r"\s*==End of OCR for page \d+==\s*==Start of OCR for page \d+==\s*")
        page_contents = separator_pattern.split(ocr_text.strip())

        for i, content in enumerate(page_contents):
            page_number = i + 1
            if content.strip():
                documents.append({
                    "filename": filename,
                    "page_number": page_number,
                    "content": content.strip()
                })
        return documents

    def ingest_pdfs(self):
        """Orchestrates the PDF ingestion process."""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[INGEST] No PDF files found in '{self.data_dir}'.")
            return

        print(f"\n[INGEST] Found {len(pdf_files)} PDF(s) to process...")
        for pdf_path in pdf_files:
            raw_ocr_output = self._ocr_pdf_with_gemini(pdf_path)
            if not raw_ocr_output:
                print(f"  [INGEST] Skipping '{pdf_path.name}' due to OCR failure.")
                continue

            docs_to_store = self._parse_gemini_ocr_output(raw_ocr_output, pdf_path.name)
            if not docs_to_store:
                print(f"  [INGEST] No content parsed from '{pdf_path.name}'.")
                continue

            print(f"  [INGEST] Storing {len(docs_to_store)} pages from '{pdf_path.name}' in the database.")
            for doc in docs_to_store:
                self.cursor.execute("""
                    INSERT OR IGNORE INTO documents (filename, page_number, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (doc["filename"], doc["page_number"], doc["content"], datetime.now(timezone.utc).isoformat()))
        
        self.cursor.execute("SELECT COUNT(*) FROM documents")
        print(f"[INGEST] Ingestion complete. Total documents in DB: {self.cursor.fetchone()[0]}")

    def create_indices(self):
        """Creates and saves both dense (FAISS) and sparse (BM25) indices."""
        print("\n[INDEX] Starting index creation process...")
        self.cursor.execute("SELECT id, content FROM documents")
        docs = self.cursor.fetchall()
        
        if not docs:
            print("[INDEX] No documents in database to index.")
            return

        print(f"[INDEX] Found {len(docs)} documents to index.")
        db_ids, contents = zip(*docs)

        # 1. Dense Index (FAISS)
        print("  [INDEX] Generating dense embeddings with Gemini...")
        try:
            result = genai.embed_content(model=self.embedding_model_name, content=contents, task_type="retrieval_document")
            embeddings = np.array(result['embedding'], dtype='float32')
            faiss.normalize_L2(embeddings)
            
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, str(self.faiss_index_path))
            
            with open(self.doc_map_path, 'wb') as f:
                pickle.dump(list(db_ids), f)
            print(f"  [INDEX] ✅ FAISS index saved with {index.ntotal} vectors.")
        except Exception as e:
            print(f"  [INDEX] ❌ Error creating FAISS index: {e}")
            return

        # 2. Sparse Index (BM25)
        print("  [INDEX] Creating sparse BM25 index...")
        tokenized_corpus = [doc.split() for doc in contents]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump(bm25, f)
        print("  [INDEX] ✅ BM25 index saved.")


def main():
    """Main function to run the full ingestion and indexing pipeline."""
    print("🚀 Starting Document Ingestion and Indexing Pipeline 🚀")
    try:
        with DocumentIngester() as ingester:
            # Step 1: Perform OCR and store results in the database
            ingester.ingest_pdfs()
            
            # Step 2: Create search indices from the database content
            ingester.create_indices()
            
        print("\n🎉 Pipeline finished successfully! Your data is ready for retrieval.")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred during the pipeline: {e}")

if __name__ == "__main__":
    main()