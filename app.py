import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="FinLogic AI", page_icon="📈", layout="wide")

# Custom CSS for a professional "Dark Finance" look
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    /* Chat bubbles */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Headers */
    h1, h2, h3 {
        color: #00e676 !important; /* Matrix Green/Finance Green */
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Input Box */
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #444;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL ARCHITECTURE (Must match Training exactly!) ---
# Copy-paste the EXACT class definitions from your Colab training script here
# (Head, MultiHeadAttention, FeedForward, Block, FinanceGPT)
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
class Head(nn.Module):
    # ... [PASTE YOUR HEAD CLASS CODE HERE] ...
    pass 
    # (For brevity in this example, I am assuming you paste the classes here)
    
class FinanceGPT(nn.Module):
    # ... [PASTE YOUR FINANCEGPT CLASS CODE HERE] ...
    pass
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_model():
    device = 'cpu' # Streamlit Cloud uses CPU
    # Initialize model with same params as training
    model = FinanceGPT() 
    # Load weights (map_location is crucial for CPU)
    try:
        model.load_state_dict(torch.load("finance_gpt_weights.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("Model weights not found. Please upload 'finance_gpt_weights.pth'.")
    return model, device

@st.cache_resource
def load_tokenizer():
    try:
        return Tokenizer.from_file("finance_tokenizer.json")
    except:
        st.error("Tokenizer not found. Upload 'finance_tokenizer.json'.")
        return None

# --- 4. APP LOGIC ---
st.title("📈 FinLogic AI")
st.caption("A Domain-Specific LLM for Finance & Analytics Students")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Parameters")
    max_tokens = st.slider("Response Length", 50, 300, 150)
    temperature = st.slider("Creativity (Temp)", 0.1, 1.0, 0.7)
    st.markdown("---")
    st.info("💡 **Tip:** Start with a concept like 'The definition of EBITDA is...' to prompt the model.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about Finance or Analytics..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Load Model & Tokenizer
        model, device = load_model()
        tokenizer = load_tokenizer()
        
        if model and tokenizer:
            with st.spinner("Analyzing financial vectors..."):
                # Encode input
                input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)
                
                # Generate (Simple loop to stream text not fully implemented for brevity, just bulk generate)
                # Note: You need to adapt the generate function in your class to accept temperature if desired
                generated_ids = model.generate(input_ids, max_new_tokens=max_tokens)[0].tolist()
                decoded_text = tokenizer.decode(generated_ids)
                
                # Clean up: Remove the prompt from the output so we don't repeat it
                response_text = decoded_text.replace(prompt, "").strip()
                
                message_placeholder.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
