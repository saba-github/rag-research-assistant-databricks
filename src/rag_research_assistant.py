import os

from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Load PDFs
# -----------------------------

VOLUME_PATH = "/Volumes/workspace/default/research_papers"

def load_documents():

    documents = []

    for filename in os.listdir(VOLUME_PATH):

        if filename.endswith(".pdf"):

            loader = PyPDFLoader(f"{VOLUME_PATH}/{filename}")

            docs = loader.load()

            documents.extend(docs)

            print(f"Loaded: {filename} -> {len(docs)} pages")

    print(f"Total pages loaded: {len(documents)}")

    return documents


# -----------------------------
# Chunk documents
# -----------------------------

def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=512,

        chunk_overlap=50,

        separators=["\n\n", "\n", ".", " "],

    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    return chunks


# -----------------------------
# Build vector store
# -----------------------------

def build_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )

    vector_store = FAISS.from_documents(chunks, embedding_model)

    print(f"Vector index ready. Total vectors: {vector_store.index.ntotal}")

    return vector_store


# -----------------------------
# Retrieve context
# -----------------------------

def retrieve_context(vector_store, question, k=4):

    docs = vector_store.similarity_search(question, k=k)

    context_parts = []

    sources = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get("source", "unknown")

        page = doc.metadata.get("page", "?")

        text = doc.page_content.strip()

        context_parts.append(

            f"[Chunk {i}] Source: {source}, Page: {page}\n{text}"

        )

        sources.append({"source": source, "page": page})

    context = "\n\n".join(context_parts)

    return context, sources, docs


# -----------------------------
# Ask RAG
# -----------------------------

def ask_rag(client, vector_store, question):

    context, sources, docs = retrieve_context(vector_store, question)

    prompt = f"""

You are a lightweight AI research assistant.

Answer the user's question using ONLY the retrieved context below.

If the context is insufficient, say so clearly.

Be concise but precise.

Retrieved context:

{context}

Question:

{question}

"""

    response = client.models.generate_content(

        model="gemini-2.5-flash",

        contents=prompt

    )

    print("\nAnswer:\n")

    print(response.text)

    print("\nSources:")

    for s in sources:

        print(f"- {s['source']} (page {s['page']})")



# -----------------------------
# Main
# -----------------------------

def main():

    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    documents = load_documents()

    chunks = chunk_documents(documents)

    vector_store = build_vector_store(chunks)

    ask_rag(client, vector_store, "What problem does QChunker try to solve?")


if __name__ == "__main__":

    main()
