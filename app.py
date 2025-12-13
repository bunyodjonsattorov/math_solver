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
    
    try:
        bytes_data = uploaded_file.getvalue()
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        
        # Detect image format from file extension
        file_extension = uploaded_file.name.split('.')[-1].lower() if uploaded_file.name else 'png'
        mime_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        mime_type = mime_type_map.get(file_extension, 'image/png')
        
        vision_llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=OPENAI_API_KEY, 
            max_tokens=1000,
            temperature=0
        )
        
        msg = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "You are a math problem transcription assistant. Transcribe this math problem EXACTLY as it appears in the image. Include all equations, numbers, and text. Do not solve the problem, only transcribe it. If the image is unclear or not a math problem, say so."
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        )
        
        response = vision_llm.invoke([msg])
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

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
                # Build context from recent messages for vague requests like "plot this"
                enhanced_input = prompt_input
                if any(keyword in prompt_input.lower() for keyword in ["plot this", "graph this", "draw this", "show this"]):
                    # Include last 2-3 messages for context
                    context_parts = []
                    for msg in st.session_state.messages[-4:-1]:  # Last 3 messages (excluding current)
                        if msg["role"] == "user":
                            context_parts.append(f"Previous question: {msg['content']}")
                        elif msg["role"] == "assistant" and "content" in msg:
                            # Extract just the math content, not the full response
                            content = msg["content"][:200]  # First 200 chars
                            context_parts.append(f"Previous answer: {content}")
                    if context_parts:
                        enhanced_input = f"{' '.join(context_parts)}\n\nCurrent request: {prompt_input}"
                
                response_data = st.session_state.agent.invoke({"input": enhanced_input})
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