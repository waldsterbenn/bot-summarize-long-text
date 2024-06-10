import os
import chromadb
from chromadb.utils import embedding_functions

max_tokens = 500  # Maximum tokens for a single chunk
overlap = 10  # Tokens to overlap between chunks to ensure continuity

embedding_func = embedding_functions.DefaultEmbeddingFunction()
embedding_func = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="data")
collection = client.get_or_create_collection(name="ls_dagbog")

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2")

# If the collection is empty, create a new one
if collection.count() == 0 | True:
    directory = "filedata"
    # List all PDF files in the directory
    files = [os.path.join(directory, f)
             for f in os.listdir(directory) if f.endswith('.md')]
    # Load a PDF document and split it into sections

    # docs = []  # Initialize an empty list to store the documents

    for file in files:
        # Assuming UnstructuredMarkdownLoader works with a single file at a time
        loader = UnstructuredMarkdownLoader(file, mode="single")
        doc = loader.load_and_split()
        collection.add(ids=[file],
                       documents=doc)

query_emb = embedding_func.encode("Ryan Graves")
question = "laser technologies"
retrived_docs = collection.query(query_texts=[question])
