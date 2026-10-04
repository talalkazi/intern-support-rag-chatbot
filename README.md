# ⚡ Intern Support AI Portal

An AI-powered, local Retrieval-Augmented Generation (RAG) web application designed to deliver real-time, context-grounded assistance for intern inquiries, organization policies, and portal guidance.

Built with **Streamlit**, **LangChain**, **FAISS**, and **Qwen2.5-0.5B-Instruct**.

---

## 🌟 Key Features

- **Context-Grounded Answers**: Utilizes local vector retrieval (FAISS) to strictly ground responses in uploaded documentation and prevent hallucinations.
- **Fast CPU Inference**: Powered by `Qwen/Qwen2.5-0.5B-Instruct` with chat template formatting optimized for quick CPU response times.
- **Dark Neon Interface**: Custom-styled Streamlit UI featuring a sleek dark charcoal theme with vibrant lime-green accent controls (`#ccff00`).
- **Transparent Source Attribution**: Expandable context viewer allowing users to inspect the exact knowledge passages used for each answer.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **RAG Orchestration**: [LangChain](https://www.langchain.com/) & `langchain-huggingface`
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model**: `Qwen/Qwen2.5-0.5B-Instruct` via Hugging Face Transformers & PyTorch

---

## 📁 Repository Structure

```text
├── app.py              # Main Streamlit user interface and RAG inference code
├── ingest.py           # Text embedding and FAISS vector index builder script
├── documents.txt       # Raw knowledge base documentation and guidelines
├── requirements.txt    # Python dependencies list
└── README.md           # Project documentation