import os
import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
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
    
    # --- 4. THE PROMPT (Specific for Tool Calling) ---
    # We do NOT use "Thought/Action" prompts here. We just give instructions.
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Cambridge A-Level Math Tutor. "
         "Your goal is to solve the user's problem using Python code. "
         "\n\n"
         "### INSTRUCTIONS:\n"
         "- You have access to a Python REPL tool. USE IT.\n"
         "- To see the output of your code, you MUST use 'print(...)'.\n"
         "- If the code errors, fix it and try again.\n"
         "- If the user asks for a graph, save it as 'graph.png' using matplotlib.\n"
         "- Once you have the answer printed, reply to the user with the final answer in LaTeX.\n"
         "- DO NOT just 'think' about it. Write and Run code."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # This is where the magic happens for tool calling
    ])

    # --- 5. THE AGENT (Modern Tool Caller) ---
    # This specific function uses OpenAI's native tool calling capability
    # It does NOT parse text for "Action:", so the error loop is impossible.
    agent = create_tool_calling_agent(llm, tools, prompt)

    # --- 6. THE EXECUTOR ---
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        # We handle errors differently now
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor
