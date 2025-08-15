#!/usr/bin/env python3
"""
A retrieval tool for performing hybrid search on an indexed knowledge base.
This tool is designed to be called by an AI agent.
"""

import os
import faiss
import pickle
import sqlite3
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

class RetrievalTool:
    """
    Performs hybrid search (Dense FAISS + Sparse BM25) on the knowledge base.
    """
    def __init__(self, db_path="db/documents.sqlite", index_dir="index"):
        """Initializes the tool by loading all necessary assets."""
        print("Initializing RetrievalTool...")
        self.db_path = db_path
        self.index_dir = index_dir
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.embedding_model = 'models/embedding-001'

        # Load all assets
        self.faiss_index = None
        self.bm25_index = None
        self.doc_map = [] # List mapping FAISS index to DB ID
        self.db_conn = None
        
        self._load_all_assets()
        print("RetrievalTool initialized successfully. Ready to search. 🚀")

    def _load_all_assets(self):
        """Loads the FAISS index, BM25 index, doc map, and connects to the DB."""
        print("  - Loading all retrieval assets...")
        
        # 1. Load FAISS index
        faiss_path = os.path.join(self.index_dir, "faiss_index.bin")
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
            print(f"    ✅ FAISS index loaded ({self.faiss_index.ntotal} vectors).")
        else:
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

        # 2. Load BM25 index
        bm25_path = os.path.join(self.index_dir, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
            print("    ✅ BM25 index loaded.")
        else:
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")

        # 3. Load FAISS-to-DB ID map
        map_path = os.path.join(self.index_dir, "doc_map.pkl")
        if os.path.exists(map_path):
            with open(map_path, 'rb') as f:
                self.doc_map = pickle.load(f)
            print(f"    ✅ Document map loaded ({len(self.doc_map)} entries).")
        else:
            raise FileNotFoundError(f"Document map not found at {map_path}")
            
        # 4. Connect to SQLite DB
        if os.path.exists(self.db_path):
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.db_conn.row_factory = sqlite3.Row # Makes rows dict-like
            print("    ✅ SQLite database connected.")
        else:
            raise FileNotFoundError(f"SQLite DB not found at {self.db_path}")

    def search(self, query: str, top_k: int = 5) -> str:
        """
        Performs hybrid search and returns a formatted string of top results.

        This is the main function to be called by an AI agent.
        """
        print(f"\nExecuting hybrid search for query: '{query}'")
        if not all([self.faiss_index, self.bm25_index, self.db_conn]):
            return "Error: Retrieval tool is not properly initialized."
        
        try:
            # 1. Dense Search (FAISS)
            query_embedding = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query" # Use 'retrieval_query' for queries
            )['embedding']
            
            query_embedding_np = np.array([query_embedding], dtype='float32')
            faiss.normalize_L2(query_embedding_np)
            
            distances, faiss_indices = self.faiss_index.search(query_embedding_np, k=20)
            dense_results = [self.doc_map[i] for i in faiss_indices[0]]
            print(f"  - Dense search returned: {dense_results[:5]}...")

            # 2. Sparse Search (BM25)
            tokenized_query = query.lower().split()
            doc_scores = self.bm25_index.get_scores(tokenized_query)
            scored_docs = list(zip(self.doc_map, doc_scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            sparse_results = [doc_id for doc_id, score in scored_docs[:20] if score > 0]
            print(f"  - Sparse search returned: {sparse_results[:5]}...")

            # 3. Reciprocal Rank Fusion (RRF) to combine results
            fused_scores = {}
            k_rrf = 60
            for rank, doc_id in enumerate(dense_results):
                if doc_id not in fused_scores: fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k_rrf + rank)
            
            for rank, doc_id in enumerate(sparse_results):
                if doc_id not in fused_scores: fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1 / (k_rrf + rank)

            reranked_results = sorted(fused_scores.keys(), key=fused_scores.get, reverse=True)
            print(f"  - Reranked (fused) results: {reranked_results[:top_k]}...")

            # 4. Fetch the top_k documents from the database
            top_ids = reranked_results[:top_k]
            if not top_ids:
                return "I couldn't find any specific information about that in the knowledge base."
            fetched_docs = self._fetch_docs_by_ids(top_ids)
            
            # 5. Format the results for the AI agent
            return self._format_results(fetched_docs)

        except Exception as e:
            print(f"An error occurred during search: {e}")
            return "Error: Could not complete the search."

    def _fetch_docs_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        """Fetches full document details from the database using their IDs."""
        if not ids:
            return []
        cursor = self.db_conn.cursor()
        placeholders = ', '.join('?' for _ in ids)
        
        # CORRECTED QUERY: Added 'id' to the SELECT statement
        query = f"SELECT id, filename, page_number, content FROM documents WHERE id IN ({placeholders})"
        
        cursor.execute(query, ids)
        
        # This map now works because 'id' is in the results
        results_map = {dict(row)['id']: dict(row) for row in cursor.fetchall()}
        
        # Re-order the results to match the reranked order
        ordered_results = [results_map[id] for id in ids if id in results_map]
        return ordered_results

    def _format_results(self, docs: List[Dict[str, Any]]) -> str:
        """Formats the retrieved documents into a single string."""
        if not docs:
            return "No relevant documents found."
            
        formatted_string = "Based on the knowledge base, here is the relevant information:\n\n"
        for i, doc in enumerate(docs, 1):
            source_info = f"Source: {doc['filename']}, Page: {doc['page_number']}"
            content_preview = doc['content'].strip()
            
            formatted_string += f"--- Result {i} ---\n"
            formatted_string += f"Context: {content_preview}\n"
            formatted_string += f"({source_info})\n\n"
        
        return formatted_string.strip()

    def close(self):
        """Closes the database connection."""
        if self.db_conn:
            self.db_conn.close()
            print("\nDatabase connection closed.")


# Example of how to use the tool
if __name__ == '__main__':
    rag_tool = None # Define in outer scope for the finally block
    try:
        # Instantiate the tool
        rag_tool = RetrievalTool()
        
        # Define a sample query
        sample_query = "What is the process for liquid biofertilizer production?"
        
        # Get the results
        results = rag_tool.search(sample_query)
        
        # Print the final formatted string
        print("\n--- Final Formatted Output for Agent ---")
        print(results)
        
    except FileNotFoundError as e:
        print(f"\nError: Could not initialize tool. An asset file is missing: {e}")
        print("Please run the 'ingest.py' script first to create the database and indices.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Cleanly close the connection if the tool was initialized
        if rag_tool:
            rag_tool.close()