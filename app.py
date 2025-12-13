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
        
        # Validate image size (OpenAI has limits)
        max_size = 20 * 1024 * 1024  # 20MB
        if len(bytes_data) > max_size:
            st.error(f"Image too large ({len(bytes_data) / 1024 / 1024:.1f}MB). Maximum size is 20MB.")
            return None
        
        # Try to validate it's actually an image using PIL
        try:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(bytes_data))
            # Convert to RGB if necessary (for formats like RGBA)
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            
            # Save to bytes in JPEG format for better compatibility
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95)
            bytes_data = output.getvalue()
            mime_type = 'image/jpeg'
        except Exception as pil_error:
            # If PIL fails, use original format
            file_extension = uploaded_file.name.split('.')[-1].lower() if uploaded_file.name else 'png'
            mime_type_map = {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp'
            }
            mime_type = mime_type_map.get(file_extension, 'image/png')
        
        base64_image = base64.b64encode(bytes_data).decode('utf-8')
        
        vision_llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=OPENAI_API_KEY, 
            max_tokens=1500,
            temperature=0
        )
        
        msg = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": (
                        "You are a math problem transcription assistant. "
                        "Look at the image carefully and transcribe the math problem EXACTLY as it appears. "
                        "Include:\n"
                        "- All mathematical expressions and equations\n"
                        "- All numbers, variables, and symbols\n"
                        "- Any text instructions or questions\n"
                        "- Formatting like fractions, exponents, etc.\n\n"
                        "Do NOT solve the problem. Only transcribe what you see in the image. "
                        "If the image contains a math problem, transcribe it completely. "
                        "If the image is unclear, blurry, or not a math problem, describe what you can see."
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"  # Request high detail for better OCR
                    }
                }
            ]
        )
        
        response = vision_llm.invoke([msg])
        result = response.content if hasattr(response, 'content') else str(response)
        
        # Validate the response
        if not result or result.strip() == "":
            return "I received the image but couldn't extract any text. Please try uploading a clearer image or type the problem manually."
        
        return result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.error(f"Error analyzing image: {str(e)}")
        # Return a helpful message instead of None
        return f"Error processing image: {str(e)}. Please try typing the problem manually or upload a different image."

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
                # Pass conversation history to agent for memory
                # Get all previous messages (excluding the current one we just added)
                conversation_history = st.session_state.messages[:-1]  # All messages except the current user message
                
                # Check if this is a follow-up question (explain, previous, etc.)
                is_followup = any(keyword in prompt_input.lower() for keyword in [
                    "explain", "previous", "above", "earlier", "before", 
                    "that solution", "the solution", "the answer", "that problem"
                ])
                
                # For follow-up questions, include more context
                if is_followup and len(conversation_history) > 0:
                    # Include more messages for context
                    conversation_history = st.session_state.messages[-8:-1]  # Last 7 messages
                
                response_data = st.session_state.agent.invoke({
                    "input": prompt_input,
                    "conversation_history": conversation_history
                })
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