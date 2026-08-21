from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIRECTORY = "./Dokumente/"
PERSIST_DIRECTORY = "./chroma_db" 
EMBEDDING_MODEL = "nomic-embed-text" 


def ingest_pdfs():
    # 2. Load all PDF documents from the directory
    print(f"Loading PDFs from '{PDF_DIRECTORY}'...")
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    raw_documents = loader.load()

    if not raw_documents:
        print("No PDF documents found.")
        return

    print(f"Loaded {len(raw_documents)} pages from PDFs.")

    # 3. Split text into chunks for better retrieval performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split documents into {len(chunks)} text chunks.")

    # 4. Initialize Ollama Embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 5. Store chunks in ChromaDB
    print("Generating embeddings and storing in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Successfully saved vector store to '{PERSIST_DIRECTORY}'.")


if __name__ == "__main__":
    ingest_pdfs()