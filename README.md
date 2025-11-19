# PDF Chat Program - README

# 📚 Chat with Multiple PDFs

This project allows you to **ask questions to multiple PDFs** using **LangChain**, **FAISS**, and **HuggingFace Transformer models**. It extracts text from PDFs, splits it into chunks, embeds the text using a sentence-transformer, and allows conversational retrieval using a large language model.

---

## Features

* Upload multiple PDFs and process them into an interactive conversational bot.
* Uses **HuggingFace pipelines locally** to avoid API limitations.
* Uses **FAISS** for efficient vector storage and retrieval.
* Maintains **chat history** for conversational context.

---

## Requirements

Python >= 3.9 (tested on **Mac M2 Air**).
Required packages are listed in `requirements.txt`.

> **Note:** FAISS may require special installation for ARM-based Macs (M1/M2). Use:
> `pip install faiss-cpu` (no GPU support needed).

---
## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Upload PDFs using the sidebar.
3. Click **Process** to create the conversational vector store.
4. Ask questions in the input box, and the bot will respond using retrieved PDF content.

---

## Code Structure

* `app.py` → Main Streamlit application.
* `htmlTemplates.py` → Contains HTML templates for chat UI.
* `requirements.txt` → Contains all Python dependencies.
* `venv/` → Optional virtual environment.

**Key Functions:**

* `get_pdf_text(pdf_docs)` → Extracts text from uploaded PDFs.
* `get_text_chunks(text)` → Splits text into manageable chunks.
* `get_vectorstore(text_chunks)` → Creates FAISS vector store from embeddings.
* `get_conversation_chain(vectorstore)` → Initializes LangChain conversational retrieval chain.
* `handle_user_input(user_question)` → Processes user input and returns the response.

---

## Why HuggingFacePipeline?

We use **HuggingFacePipeline** because it allows us to load transformer models locally using the `transformers` library without relying on a HuggingFaceHub endpoint.
This avoids errors like `Task not specified` and `StopIteration` which occur with HuggingFaceHub or HuggingFaceEndpoint when using unsupported or cloud-only models.

**Tokenizer Usage:**

The tokenizer is used internally by the HuggingFace pipeline to convert text input into tokens the model can understand. It ensures proper text encoding for the transformer model and controls input length (`max_length`) and output generation (`max_new_tokens`).

---

## Common Issues & Troubleshooting

### 1️⃣ HuggingFaceHub errors:

* **Error:** `Task not specified` or `NotImplementedError`
* **Solution:** Use **HuggingFacePipeline with a local transformers pipeline** instead.

### 2️⃣ HuggingFaceEndpoint errors:

* **Error:** `StopIteration` or `provider_helper` issues
* **Solution:** Use **local HuggingFace pipeline** (no cloud required).

### 3️⃣ HuggingFacePipeline errors:

### 4️⃣ FAISS installation on M1/M2 Macs:

* **Error:** Compilation fails on ARM architecture
* **Solution:** Install prebuilt CPU version:

### 5️⃣ Cache issues / large downloads:

* Transformers models and embeddings are cached in:
  `~/.cache/huggingface/`
* To clear space:

```bash
rm -rf ~/.cache/huggingface/
```

---
## References

* [LangChain Documentation](https://python.langchain.com/docs/)
* [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/index)
* [FAISS GitHub](https://github.com/facebookresearch/faiss)

---