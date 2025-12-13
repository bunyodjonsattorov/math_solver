import streamlit as st
import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent import get_math_agent

# --- Page Config ---
st.set_page_config(page_title="Cambridge Math AI", page_icon="🎓", layout="wide")

# --- API Key Setup for Vision ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except:
        pass

# --- Helper Functions ---
def analyze_image(uploaded_file):
    """Uses GPT-4o Vision to transcribe image."""
    if not OPENAI_API_KEY:
        st.error("API Key missing.")
        return None
        
    bytes_data = uploaded_file.getvalue()
    base64_image = base64.b64encode(bytes_data).decode('utf-8')
    
    vision_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, max_tokens=500)
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Transcribe this math problem EXACTLY. Do not solve it."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    return vision_llm.invoke([msg]).content

def process_and_display(prompt_input):
    """Runs the agent and updates UI."""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🧠 Thinking & Plotting..."):
            try:
                response_data = st.session_state.agent.invoke({"input": prompt_input})
                final_answer = response_data['output']
                # Note: create_python_agent may not return intermediate_steps
                steps = response_data.get('intermediate_steps', [])
                
                # 1. Show Reasoning
                if steps:
                    with st.expander("View Python Logic"):
                        for step in steps:
                            try:
                                action = step[0]
                                observation = step[1]
                                st.code(action.tool_input, language="python")
                                st.text(f"Output: {observation}")
                            except:
                                pass
                
                # 2. Show Answer
                message_placeholder.markdown(final_answer)
                
                # 3. Check for Graph
                image_data = None
                if os.path.exists("graph.png"):
                    st.image("graph.png", caption="Generated Plot")
                    # Read image to save in history
                    with open("graph.png", "rb") as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    os.remove("graph.png")  # Clean up
                
                # 4. Save to History
                msg_data = {
                    "role": "assistant", 
                    "content": final_answer,
                    "steps": steps
                }
                if image_data:
                    msg_data["image"] = image_data
                
                st.session_state.messages.append(msg_data)
                
            except Exception as e:
                st.error(f"Agent Error: {e}")

# --- Main UI ---
st.title("🎓 Cambridge A-Level Math Solver")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = get_math_agent()

# Sidebar
with st.sidebar:
    st.header("Upload Problem")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.pop("last_processed_file", None) # Reset file memory
        st.rerun()

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Check if there is an image saved in this message
        if "image" in msg and msg["image"]:
            st.image(base64.b64decode(msg["image"]), caption="Generated Plot")
        
        if "steps" in msg and msg["steps"]:
            with st.expander("View Python Logic"):
                for step in msg["steps"]:
                    try:
                        st.code(step[0].tool_input, language="python")
                        st.text(f"Output: {step[1]}")
                    except:
                        pass

# Handle Image Upload
if uploaded_file:
    # Check if we already processed this exact file
    if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != uploaded_file.name:
        text = analyze_image(uploaded_file)
        if text:
            process_and_display(f"I uploaded an image. Problem: {text}")
            st.session_state.last_processed_file = uploaded_file.name
            st.rerun() # Refresh to show new state

# Handle Text Input
if prompt := st.chat_input("Type your math problem..."):
    process_and_display(prompt)