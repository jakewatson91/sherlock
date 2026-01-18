---
title: Sherlock RAG
emoji: 🕵️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Sherlock RAG 🕵️

A specialized Chainlit application implementing a Retrieval-Augmented Generation (RAG) pipeline. This assistant is designed to query private project documentation and resumes to provide context-aware answers using high-performance LLMs.

## 🏗️ Architecture

This project follows a decoupled data-and-code architecture:

- **Application Logic**: Hosted on GitHub and deployed to Hugging Face Spaces via Docker.
- **Knowledge Base**: Private PDF documents stored in a separate Hugging Face Dataset (`jakewatson91/sherlock-rag-docs`).
- **Sync Mechanism**: The app uses `huggingface_hub` to sync documents at runtime, bypassing Git LFS limitations and keeping the code repository lightweight.

## 🛠️ Tech Stack

- **UI/UX**: [Chainlit](https://docs.chainlit.io/)
- **Orchestration**: [LangChain](https://python.langchain.com/)
- **LLM**: Moonshot AI (Kimi-k2) via [Groq](https://groq.com/)
- **Embeddings**: Google Generative AI (`text-embedding-004`)
- **Data**: From HuggingFace Dataset `huggingface_hub` (Snapshot Download)

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A Hugging Face **Write** Token
- API Keys for:
  - Groq (Moonshot AI)
  - Google Generative AI (Embeddings)

### Environment Variables

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_write_token
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```
