# AI Research Assistant (RAG Prototype)

A lightweight Retrieval-Augmented Generation (RAG) prototype built on Databricks Free Edition.

This project demonstrates how academic PDFs can be ingested, chunked, embedded, indexed with FAISS, and queried through a source-grounded Gemini workflow.

## Architecture

PDF ingestion → text chunking → embeddings → FAISS vector search → Gemini grounded answering

## Technologies

- Databricks Free Edition
- LangChain
- Sentence Transformers
- FAISS
- Google Gemini

## What it does

The system:

1. Loads academic PDF papers from a Databricks volume
2. Splits documents into smaller text chunks
3. Generates semantic embeddings
4. Builds a FAISS vector index
5. Retrieves the most relevant chunks for a question
6. Uses Gemini to generate grounded answers based only on retrieved context
7. Prints source references and page numbers

## Example questions

- What problem does QChunker try to solve?
- How does Legal-DC evaluate retrieval-augmented generation for legal documents?
- What is the main idea behind evolving memory in LLM agents?

## Repository structure

```text
rag-research-assistant-databricks/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    └── rag_research_assistant.py
