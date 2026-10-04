import streamlit as st

st.set_page_config(
    page_title="Intern Support AI",
    page_icon="⚡",
    layout="wide"
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.markdown("""
<style>
    .stApp {
        background-color: #0b0c10;
        color: #ffffff;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .badge-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .neon-badge {
        background-color: #151a11;
        color: #ccff00;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        border: 1px solid rgba(204, 255, 0, 0.3);
        box-shadow: 0 0 12px rgba(204, 255, 0, 0.15);
        text-transform: uppercase;
    }

    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .neon-text {
        color: #ccff00;
    }
    .sub-caption {
        text-align: center;
        color: #9aa0a6;
        font-size: 0.95rem;
        margin-bottom: 35px;
    }

    .stSelectbox label, .stTextArea label, h3 {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    div[data-baseweb="select"] > div, textarea {
        background-color: #14171d !important;
        border: 1px solid #262b36 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div:focus-within, textarea:focus {
        border-color: #ccff00 !important;
        box-shadow: 0 0 8px rgba(204, 255, 0, 0.2) !important;
    }

    div.stButton > button {
        background: #ccff00 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 0 15px rgba(204, 255, 0, 0.25) !important;
    }
    
    div.stButton > button:hover {
        background: #b3ff00 !important;
        transform: translateY(-1px);
        box-shadow: 0 0 20px rgba(204, 255, 0, 0.4) !important;
        color: #000000 !important;
    }

    div.stAlert {
        background-color: #121812 !important;
        border: 1px solid #233816 !important;
        color: #ccff00 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] {
        background-color: #14171d !important;
        border: 1px solid #232833 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="badge-container"><span class="neon-badge">INTERN CHATBOT SERVICE</span></div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">The Next-Gen Support Portal for <span class="neon-text">Smart Assistance</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-caption">AI-powered guidance and instant intern support</p>', unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False,
        return_full_text=False
    )
    
    return vectorstore, tokenizer, pipe

vectorstore, tokenizer, pipe = load_rag_pipeline()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Query Parameters")
    category = st.selectbox(
        "Select Category",
        [
            "General Support & Policies",
            "Submissions & Portal Access",
            "Edge Cases & Specific Queries",
            "Off-Topic / Out-of-Bounds"
        ]
    )
    
    user_query = st.text_area(
        "Enter Your Inquiry",
        placeholder="Type your question here...",
        height=120
    )
    
    submit_button = st.button("Get Response")

with col2:
    st.subheader("Retrieved Answer & Context")
    
    if submit_button and user_query:
        with st.spinner("Analyzing knowledge base..."):
            docs = vectorstore.similarity_search(user_query, k=2)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            system_instruction = (
                "You are a helpful, professional AI assistant for the Intern Support Portal. "
                "Use ONLY the provided context to answer the user's question directly and concisely. "
                "If the user asks about something completely unrelated to intern guidelines (e.g. recipes, order tracking), reply politely: "
                "'I am specialized in intern portal support and guidelines. I don't have information on that topic.'"
            )
            
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
            ]
            
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            output = pipe(prompt_text)
            clean_response = output[0]["generated_text"].strip()

            st.markdown("**Assistant Output:**")
            st.success(clean_response)
            
            with st.expander("View Matching Knowledge Sources"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(doc.page_content)