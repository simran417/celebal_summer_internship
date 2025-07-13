import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

# 2. CONFIGURATION - Adjust these values:
DATA_PATH = r"c:\Users\hp\OneDrive\Documents\Training Dataset.csv"  # Update path
MODEL_NAME = "google/flan-t5-base"  # Using smaller base model
MAX_CHUNK_SIZE = 256  # Reduced from 1000 to handle memory better
BATCH_SIZE = 4  # Smaller batches for memory-constrained systems

def memory_optimized_rag():
    try:
        # 3. MEMORY OPTIMIZED DOCUMENT LOADING
        print("🔄 Loading and splitting documents...")
        loader = CSVLoader(
            file_path=DATA_PATH,
            encoding='utf-8',
            csv_args={'fieldnames': ['text']}  # Adjust based on your CSV structure
        )
        documents = loader.load()
        
        # 4. SMARTER TEXT SPLITTING
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(documents)
        print(f"✅ Split {len(documents)} documents into {len(texts)} chunks")
        
        # 5. MEMORY-EFFICIENT EMBEDDINGS
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU to save GPU memory
            encode_kwargs={
                'batch_size': BATCH_SIZE,
                'normalize_embeddings': True
            }
        )
        
        # 6. PROCESS IN BATCHES TO AVOID OOM
        print("🔄 Creating vector store (batched processing)...")
        db = FAISS.from_documents(texts[:100], embeddings)  # Start with first 100
        for i in range(100, len(texts), 100):
            db.add_documents(texts[i:i+100])
        print("✅ Vector store created successfully")
        
        # 7. MEMORY-OPTIMIZED LLM
        print("🔄 Loading LLM with memory optimizations...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=MAX_CHUNK_SIZE,
            device="cpu",  # Force CPU if memory is tight
            batch_size=BATCH_SIZE
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✅ LLM initialized successfully")
        
        # 8. CREATE RAG CHAIN
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",  # Better for large docs than "stuff"
            retriever=db.as_retriever(
                search_kwargs={'k': 3}  # Return fewer docs to save memory
            ),
            return_source_documents=False  # Skip to save memory
        )
        
        # 9. TEST WITH MEMORY SAFETY
        print("\n🔍 Testing with sample query...")
        query = "What is RAG?"
        result = qa_chain.invoke({"query": query})
        print("Answer:", result["result"])
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nAdditional troubleshooting:")
        print("- Try reducing MAX_CHUNK_SIZE further (e.g., to 128)")
        print("- Close other memory-intensive applications")
        print("- Consider using Google Colab with GPU if problems persist")

if __name__ == "__main__":
    memory_optimized_rag()