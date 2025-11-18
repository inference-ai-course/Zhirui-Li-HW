"""
RAG Pipeline for arXiv Papers - WITH SSL FIX
Complete implementation with PDF extraction, chunking, embedding, and search
INCLUDES SSL CERTIFICATE FIX FOR WINDOWS
"""

import os
import pickle
import ssl
import certifi
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import arxiv

# FIX SSL CERTIFICATE ISSUES
ssl._create_default_https_context = ssl._create_unverified_context
print("✅ SSL verification configured for arXiv downloads\n")


class RAGPipeline:
    """Complete RAG pipeline for processing and searching academic papers"""
    
    def __init__(self, 
                 papers_dir: str = "papers",
                 processed_dir: str = "processed",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline
        
        Args:
            papers_dir: Directory to store downloaded PDFs
            processed_dir: Directory to store processed data
            model_name: Sentence transformer model name
        """
        self.papers_dir = Path(papers_dir)
        self.processed_dir = Path(processed_dir)
        self.papers_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        self.chunks = []
        self.chunk_metadata = []  # Store paper ID, chunk index, etc.
        self.index = None
        
    def download_papers(self, query: str, max_results: int = 50):
        """
        Download papers from arXiv with SSL fix
        
        Args:
            query: Search query for arXiv
            max_results: Maximum number of papers to download
        """
        print(f"Searching arXiv for: '{query}'")
        
        # Use the newer Client API to avoid deprecation warning
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        downloaded = 0
        errors = 0
        
        for paper in client.results(search):
            try:
                paper_id = paper.get_short_id()
                pdf_path = self.papers_dir / f"{paper_id}.pdf"
                
                if pdf_path.exists():
                    print(f"  ✓ Already exists: {paper_id}")
                    downloaded += 1
                    continue
                
                # Download with SSL context that accepts self-signed certificates
                paper.download_pdf(filename=str(pdf_path))
                print(f"  ✓ Downloaded: {paper_id} - {paper.title[:50]}...")
                downloaded += 1
                
            except Exception as e:
                errors += 1
                print(f"  ✗ Error downloading {paper.get_short_id()}: {str(e)[:80]}")
                # Continue trying other papers
                continue
                
        print(f"\n✅ Successfully downloaded/found: {downloaded} papers")
        if errors > 0:
            print(f"⚠️  Failed to download: {errors} papers (continuing with what we have)")
        return downloaded
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as a single string
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page in doc:
                page_text = page.get_text()
                # Clean page text (remove headers/footers if needed)
                pages.append(page_text)
            
            full_text = "\n".join(pages)
            doc.close()
            return full_text
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, 
                   text: str, 
                   max_tokens: int = 512, 
                   overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (approximate)
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple word-based chunking (words ≈ tokens for English)
        words = text.split()
        chunks = []
        step = max_tokens - overlap
        
        for i in range(0, len(words), step):
            chunk = words[i:i + max_tokens]
            chunk_text = " ".join(chunk)
            if len(chunk_text.strip()) > 50:  # Minimum chunk size
                chunks.append(chunk_text)
        
        return chunks
    
    def process_all_papers(self, max_tokens: int = 512, overlap: int = 50):
        """
        Process all PDFs in papers directory
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Overlap between chunks
        """
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("No PDF files found. Please download papers first.")
            return
        
        print(f"\nProcessing {len(pdf_files)} papers...")
        
        all_chunks = []
        all_metadata = []
        
        for idx, pdf_path in enumerate(pdf_files):
            paper_id = pdf_path.stem
            print(f"  [{idx+1}/{len(pdf_files)}] Processing {paper_id}...")
            
            # Extract text
            text = self.extract_text_from_pdf(str(pdf_path))
            
            if not text:
                print(f"    ⚠️  Warning: No text extracted from {paper_id}")
                continue
            
            # Chunk text
            chunks = self.chunk_text(text, max_tokens, overlap)
            print(f"    ✓ Created {len(chunks)} chunks")
            
            # Store chunks and metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "paper_id": paper_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                })
        
        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        
        print(f"\n✅ Total chunks created: {len(self.chunks)}")
    
    def build_index(self):
        """
        Generate embeddings and build FAISS index
        """
        if not self.chunks:
            print("No chunks to index. Please process papers first.")
            return
        
        print(f"\nGenerating embeddings for {len(self.chunks)} chunks...")
        print("This may take a few minutes...")
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch)}/{len(self.chunks)} chunks")
        
        embeddings = np.vstack(embeddings)
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Build FAISS index
        print("Building FAISS index...")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✅ Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of results with chunks and metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Please build index first.")
        
        # Embed query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k
        )
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Safety check
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "distance": float(distances[0][i]),
                    "rank": i + 1
                })
        
        return results
    
    def save(self):
        """Save processed data and index"""
        print("\nSaving processed data...")
        
        # Save chunks and metadata
        with open(self.processed_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        with open(self.processed_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(self.processed_dir / "faiss_index.bin"))
        
        print("✅ Data saved successfully!")
    
    def load(self):
        """Load processed data and index"""
        print("\nLoading processed data...")
        
        # Load chunks and metadata
        chunks_path = self.processed_dir / "chunks.pkl"
        metadata_path = self.processed_dir / "metadata.pkl"
        index_path = self.processed_dir / "faiss_index.bin"
        
        if not chunks_path.exists():
            print("No processed data found. Please process papers first.")
            return False
        
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        with open(metadata_path, "rb") as f:
            self.chunk_metadata = pickle.load(f)
        
        # Load FAISS index
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        print(f"✅ Loaded {len(self.chunks)} chunks and index with {self.index.ntotal} vectors")
        return True


# FastAPI Application
app = FastAPI(title="RAG Search API", description="Search academic papers using RAG")

# Global pipeline instance
pipeline = None


class SearchRequest(BaseModel):
    query: str
    k: int = 3


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    global pipeline
    pipeline = RAGPipeline()
    
    # Try to load existing processed data
    if not pipeline.load():
        print("No processed data found. Please run the processing pipeline first.")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Search API",
        "endpoints": {
            "/search": "Search for relevant passages",
            "/stats": "Get index statistics"
        }
    }


@app.get("/search")
async def search(q: str, k: int = 3):
    """
    Search endpoint
    
    Args:
        q: Query string
        k: Number of results (default: 3)
    """
    if pipeline is None or pipeline.index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")
    
    try:
        results = pipeline.search(q, k)
        return {
            "query": q,
            "num_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/stats")
async def stats():
    """Get index statistics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "total_chunks": len(pipeline.chunks) if pipeline.chunks else 0,
        "index_size": pipeline.index.ntotal if pipeline.index else 0,
        "papers_directory": str(pipeline.papers_dir),
        "processed_directory": str(pipeline.processed_dir)
    }


if __name__ == "__main__":
    # CLI for building the pipeline
    import sys
    
    print("=" * 60)
    print("RAG Pipeline Builder (WITH SSL FIX)")
    print("=" * 60)
    
    pipeline = RAGPipeline()
    
    # Check if we should load existing data
    if pipeline.processed_dir.exists() and list(pipeline.processed_dir.glob("*.pkl")):
        load = input("\nFound existing processed data. Load it? (y/n): ").lower()
        if load == 'y':
            pipeline.load()
            print("\n✅ Pipeline ready! Starting FastAPI server...")
            print("Run: uvicorn rag_pipeline_fixed:app --reload")
            sys.exit(0)
    
    # Step 1: Download papers
    print("\n" + "=" * 60)
    print("STEP 1: Download Papers (SSL FIXED)")
    print("=" * 60)
    
    download = input("Download papers from arXiv? (y/n): ").lower()
    
    if download == 'y':
        query = input("Enter search query (default: 'machine learning'): ").strip()
        if not query:
            query = "machine learning"
        
        num_papers = input("Number of papers (default: 50): ").strip()
        num_papers = int(num_papers) if num_papers else 50
        
        downloaded = pipeline.download_papers(query, num_papers)
        
        if downloaded == 0:
            print("\n❌ No papers downloaded. Options:")
            print("1. Check your internet connection")
            print("2. Try again")
            print("3. Add PDFs manually to 'papers/' folder")
            sys.exit(1)
    
    # Check if we have papers
    pdf_count = len(list(pipeline.papers_dir.glob("*.pdf")))
    if pdf_count == 0:
        print("\n❌ No PDFs found. Please add PDF files to the 'papers' directory.")
        sys.exit(1)
    
    print(f"\n✅ Found {pdf_count} PDF files")
    
    # Step 2: Process papers
    print("\n" + "=" * 60)
    print("STEP 2: Process Papers")
    print("=" * 60)
    
    process = input("Process papers (extract text and chunk)? (y/n): ").lower()
    
    if process == 'y':
        pipeline.process_all_papers(max_tokens=512, overlap=50)
    else:
        print("Skipping processing. Exiting.")
        sys.exit(0)
    
    # Step 3: Build index
    print("\n" + "=" * 60)
    print("STEP 3: Build Index")
    print("=" * 60)
    
    build = input("Build FAISS index? (y/n): ").lower()
    
    if build == 'y':
        pipeline.build_index()
    else:
        print("Skipping index building. Exiting.")
        sys.exit(0)
    
    # Step 4: Test search
    print("\n" + "=" * 60)
    print("STEP 4: Test Search")
    print("=" * 60)
    
    test = input("Test the search? (y/n): ").lower()
    
    if test == 'y':
        query = input("Enter test query: ").strip()
        if query:
            results = pipeline.search(query, k=3)
            
            print(f"\n✅ Top 3 results for: '{query}'")
            print("=" * 60)
            
            for result in results:
                print(f"\nRank {result['rank']} (Distance: {result['distance']:.4f})")
                print(f"Paper: {result['metadata']['paper_id']}")
                print(f"Chunk: {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}")
                print("-" * 60)
                print(result['chunk'][:300] + "...")
                print()
    
    # Step 5: Save data
    print("\n" + "=" * 60)
    print("STEP 5: Save Data")
    print("=" * 60)
    
    pipeline.save()
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print("\nTo start the FastAPI server, run:")
    print("  uvicorn rag_pipeline_fixed:app --reload")
    print("\nThen visit:")
    print("  http://localhost:8000/search?q=your+query&k=3")
