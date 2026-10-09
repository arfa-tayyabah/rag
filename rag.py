!pip install langchain langchain-community qdrant-client pdfminer.six
!pip install langchain-qdrant sentence-transformers langchain-text-splitters langchain-huggingface

from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

PDF_PATH = "/content/Student_HandBook.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

loader = PDFMinerLoader(PDF_PATH)
pdf_content = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)
documents = text_splitter.split_documents(pdf_content)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

qdrant = QdrantVectorStore.from_documents(
    documents,
    embeddings,
    path="/tmp/local_qdrant",
    collection_name="my_documents",
)

found_docs = qdrant.similarity_search("What is the message from founder", k=3)
for i, doc in enumerate(found_docs, 1):
    print(f"--- Result {i} ---")
    print(doc.page_content)
