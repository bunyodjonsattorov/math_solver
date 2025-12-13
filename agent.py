import os
import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool

# --- 1. SETUP API KEY ---
# Try to get key from Streamlit secrets, otherwise environment variable
def _get_api_key():
    """Get API key from secrets or environment."""
    OPENAI_API_KEY = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Please set it in secrets or environment.")
    
    return OPENAI_API_KEY

def get_math_agent():
    # --- 2. THE BRAIN (LLM) ---
    OPENAI_API_KEY = _get_api_key()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # --- 3. THE TOOLS ---
    tools = [PythonREPLTool()]
    
    # --- 4. THE PROMPT (Optimized for ReAct to avoid parsing errors) ---
    # Using a cleaner prompt format that works better with ReAct
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Cambridge A-Level Math Tutor. "
         "Your goal is to solve the user's problem using Python code.\n\n"
         "You have access to a Python REPL tool. Always use it to solve problems.\n"
         "To see output, use print(). For graphs, save to 'graph.png' using matplotlib.\n"
         "After getting the answer, provide it in LaTeX format."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # --- 5. CREATE REACT AGENT (More compatible) ---
    # This is more widely available than create_tool_calling_agent
    agent = create_react_agent(llm, tools, prompt)

    # --- 6. THE EXECUTOR WITH BETTER ERROR HANDLING ---
    def handle_parsing_error(error):
        """Custom error handler to prevent loops."""
        return "I need to use the Python tool to solve this. Let me try again with the correct format."

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=handle_parsing_error,
        max_iterations=10,
        return_intermediate_steps=True
    )
    
    return agent_executor
