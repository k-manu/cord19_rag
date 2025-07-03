import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="COVID-19 Research RAG Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color: #000000 !important;
}
.user-message {
    background-color: #e1f5fe;
    border-left: 4px solid #1f77b4;
    color: #000000 !important;
}
.bot-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
    color: #000000 !important;
}
.source-box {
    background-color: #f5f5f5;
    padding: 0.5rem;
    border-radius: 5px;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #333333 !important;
}
/* Force dark text for chat content only */
.chat-message .stMarkdown, .chat-message .stMarkdown p, .chat-message .stMarkdown div {
    color: #000000 !important;
}
/* Ensure chat input text is visible */
.stChatInput input {
    color: #000000 !important;
}
/* Make intro description text white */
.intro-text {
    color: #ffffff !important;
}
/* Target the description paragraph specifically */
.main .block-container .stMarkdown p {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_chain():
    """Initialize the RAG chain with cached vectorstore"""
    try:
        # Initialize embeddings
        embedding_model = OpenAIEmbeddings()
        
        # Load existing vectorstore
        vectorstore = Chroma(
            persist_directory="chroma_cord19",
            embedding_function=embedding_model
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Initialize LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # Create prompt template
        template = """You are a helpful assistant with access to COVID-19 medical research.

Answer the question using the following context from recent research papers:
{context}

Question: {question}
Answer: Please provide a comprehensive answer based on the research context. If the context doesn't contain relevant information, please say so clearly."""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever
    
    except Exception as e:
        st.error(f"Error initializing RAG chain: {str(e)}")
        return None, None

def main():
    st.markdown('<h1 class="main-header">🧬 COVID-19 Research RAG Chatbot</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="intro-text">
    This chatbot can answer questions about COVID-19 based on research papers from the CORD-19 dataset. 
    Ask any questions about COVID-19 symptoms, treatments, neurological effects, vaccines, or other research topics.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
        st.session_state.retriever = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Check if API key is available
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Please ensure your .env file contains OPENAI_API_KEY")
        
        st.markdown("---")
        st.header("📚 About")
        st.info("""
        This chatbot uses:
        - **CORD-19 Dataset**: 2000 recent COVID-19 research papers
        - **Vector Search**: Chroma vectorstore with OpenAI embeddings
        - **LLM**: GPT-3.5-turbo for generating responses
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize RAG chain
    if st.session_state.rag_chain is None:
        with st.spinner("🔄 Loading vectorstore and initializing RAG chain..."):
            rag_chain, retriever = initialize_rag_chain()
            if rag_chain:
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.success("✅ RAG chain initialized successfully!")
            else:
                st.error("❌ Failed to initialize RAG chain. Please check your configuration.")
                return
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about COVID-19 research..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {prompt}</div>', unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                # Get response from RAG chain
                response = st.session_state.rag_chain.invoke(prompt)
                
                # Get source documents for transparency
                source_docs = st.session_state.retriever.get_relevant_documents(prompt)
                sources = []
                for doc in source_docs:
                    title = doc.metadata.get("title", "Unknown Title")
                    publish_time = doc.metadata.get("publish_time", "Unknown Date")
                    content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    sources.append(f"**{title}** ({publish_time})\n{content_preview}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
                
                # Display assistant response
                st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {response}</div>', unsafe_allow_html=True)
                
                # Display sources
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(f'<div class="source-box">{source}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {error_message}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 