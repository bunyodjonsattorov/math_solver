import streamlit as st
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent import get_math_agent
import os

# Key comes from env/secrets; agent.py also loads it for the main LLM.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Page Config ---
st.set_page_config(
    page_title="Cambridge Math AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for layout/brand ---
st.markdown("""
<style>
    .stChatMessage {
        background-color: #f0f2f6; 
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2c3e50;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4a90e2;
        text-align: center;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper: transcribe an uploaded image ---
def analyze_image(uploaded_file):
    """
    Uses GPT-4o Vision to read the math problem from the image.
    """
    # Encode image to base64
    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode('utf-8')
    
    # Vision model
    vision_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, max_tokens=500)
    
    # Ask it to transcribe
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Look at this math problem. Transcribe it EXACTLY into text. If there is a diagram, describe it briefly so a solver can understand it. Do not solve it, just transcribe it."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    response = vision_llm.invoke([msg])
    return response.content

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=50)
    st.title("Math Tools")
    st.markdown("---")
    
    # Image Uploader in Sidebar
    uploaded_file = st.file_uploader("📸 Upload a Math Problem", type=["jpg", "png", "jpeg"])
    st.caption("Upload a screenshot; we transcribe with GPT-4o, then solve.")
    
    st.markdown("---")
    st.markdown("### 📚 Topics Supported")
    st.markdown("""
    - Pure Mathematics 1, 2 & 3
    - Probability & Statistics
    - Mechanics
    - Differential Equations
    """)
    st.markdown("### 🧭 How to use")
    st.markdown("- Upload an image or type a question.\n- Agent shows code under \"Thinking Process\".\n- Final answer is rendered with LaTeX.")
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Layout ---
st.markdown('<div class="main-header">Cambridge A-Level Math Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by GPT-4o & Python REPL</div>', unsafe_allow_html=True)

# --- Initialize Agent ---
if "agent" not in st.session_state:
    st.session_state.agent = get_math_agent()

# --- Initialize History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat ---
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        # If there are steps stored in this message, display them
        if "steps" in message and message["steps"]:
            with st.expander("Show Thinking Process (Python Code)"):
                for step in message["steps"]:
                    st.code(step[0].tool_input, language="python")
                    st.text(f"Output: {step[1]}")

# --- Helper to process response ---
def process_response(user_input):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🧠 Analyzing problem & Calculating..."):
            try:
                response_data = st.session_state.agent.invoke({"input": user_input})
                final_answer = response_data['output']
                intermediate_steps = response_data.get('intermediate_steps', [])
                
                # Display nicely
                message_placeholder.markdown(final_answer)
                
                # Show steps in an expander immediately
                if intermediate_steps:
                    with st.expander("Show Thinking Process (Python Code)"):
                        for step in intermediate_steps:
                            # step is a tuple: (AgentAction, observation)
                            st.code(step[0].tool_input, language="python")
                            st.text(f"Output: {step[1]}")
                
                # Save to history with steps
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_answer,
                    "steps": intermediate_steps
                })
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- Logic: Handle Image Upload ---
if uploaded_file:
    with st.spinner("👀 Reading the image..."):
        # We only want to process the image ONCE.
        if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != uploaded_file.name:
            
            # 1. Analyze Image
            transcribed_text = analyze_image(uploaded_file)
            st.toast("Image read successfully!", icon="✅")
            
            # 2. Add to chat as if USER typed it
            prompt_text = f"I have uploaded an image. Here is the problem: {transcribed_text}"
            st.session_state.messages.append({"role": "user", "content": prompt_text})
            with st.chat_message("user"):
                st.markdown(prompt_text)
            
            # 3. Process Response
            process_response(prompt_text)
            
            st.session_state.last_processed_file = uploaded_file.name # Mark as done

# --- Logic: Handle Text Input ---
if prompt := st.chat_input("Enter your complex math problem here..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Agent Response
    process_response(prompt)
