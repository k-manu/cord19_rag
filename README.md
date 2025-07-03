# COVID-19 Research RAG Chatbot

A **Streamlit-powered chatbot** built using **LangChain**, **OpenAI GPT-3.5 Turbo**, and **Chroma vector store**, enabling you to explore the most recent **COVID-19 research papers** with Retrieval-Augmented Generation (RAG). Tracing and observability are integrated with **LangSmith**.

---

## Features

* 🤖 **Ask natural-language questions** about COVID-19 backed by scientific papers
* 📚 Uses the **2,000 most recent papers** from the CORD-19 dataset
* 🔍 **Chroma vector store** with OpenAI Embeddings for semantic search
* 🧠 Powered by **GPT-3.5 Turbo** for response generation
* 🧾 **Transparent source context** for each answer (coming soon!)
* 💬 Streamlit interface with persistent session history
* 📊 **LangSmith observability** for tracing the full RAG pipeline
* 🧩 Built with **LangChain** for modular, composable AI workflows

---

## Project Structure

```
cord19_rag/
├── app.py                # Streamlit UI + RAG logic
├── requirements.txt      # Required Python packages
├── .env                  # Environment variables for API keys
├── chroma_cord19/        # Persisted Chroma vectorstore
└── README.md             # Project documentation
```

---

## Setup Instructions

### 1.  Install Dependencies

```bash
pip install -r requirements.txt
```

### 2.  Configure Environment Variables

Create a `.env` file with your credentials:

```env
OPENAI_API_KEY=your_actual_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_key  # Optional but recommended
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

> You can create a free LangSmith account at: [https://smith.langchain.com](https://smith.langchain.com)

### 3.  Run the App

```bash
streamlit run app.py
```

---

##  How It Works

* The app uses **LangChain's `Retriever-LLM` chain**, where your question is:

  1. Embedded via **OpenAI Embeddings**
  2. Searched against **Chroma vector DB** (pre-built from full-text CORD-19 papers)
  3. Passed as context into a **LangChain `ChatPromptTemplate`**
  4. Answered by **GPT-3.5 Turbo**, then returned to you in Streamlit

* All RAG pipeline activity is optionally **traced in LangSmith**, including:

  * Retrieved documents
  * Final answer
  * Prompt and token usage breakdown

---

## Example Questions

Try asking:

* “What are the neurological effects of COVID-19?”
* “What treatments have been proven effective in clinical trials?”
* “Is there a connection between COVID-19 and autoimmune diseases?”
* “What long-term effects does COVID-19 have on young adults?”

---

##  About LangChain

This project is powered by [LangChain](https://docs.langchain.com), a framework for building language model applications using modular and composable components. LangChain enables:

* Flexible RAG pipelines
* Plug-and-play vector stores and LLMs
* Structured chains and observability
* Tracing and debugging with LangSmith

---

##  About LangSmith (Optional but Powerful), use LangFuse if using extensively(cuz its open-source)

[LangSmith](https://smith.langchain.com) is LangChain's developer platform for monitoring and evaluating LLM applications. In this project, it helps you:

* Visualize your full RAG chain execution
* Debug retrieval and generation steps
* Analyze token usage and prompt formatting

Once enabled via `.env`, LangSmith works silently behind the scenes — and can be viewed at [smith.langchain.com](https://smith.langchain.com).It logs all the traces and makes it easier to debug.

---

## Troubleshooting

* **“OpenAI API Key not found”**
  → Make sure `.env` exists and is properly loaded

* **“Failed to load vectorstore”**
  → Ensure `chroma_cord19/` exists and contains your previously embedded documents

* **First query is slow**
  → This is normal while Chroma initializes; later queries are faster

---

## Future Enhancements

* Show source paper titles + links in responses
* Add multi-query RAG or HyDE-style generation(for fetching even more relevant documents)
* Let users download responses as PDFs

---

## Acknowledgements

* **CORD-19 Dataset** by the Allen Institute for AI
* **LangChain** for enabling modular LLM pipelines
* **OpenAI** for embeddings + LLMs
* **ChromaDB** for fast, local vector search
* **LangSmith** for advanced observability

---