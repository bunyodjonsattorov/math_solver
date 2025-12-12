import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool

# Load API key (Streamlit secrets first, then env)
OPENAI_API_KEY = None
try:
    # 1. Try Streamlit Secrets (Production)
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not OPENAI_API_KEY:
    # 2. Try Environment Variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in secrets, config.py, or environment variables.")

def get_math_agent():
    """Build a math agent that uses Python for the heavy lifting."""
    # 1. Setup the Brain
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # 2. Setup Tools
    tools = [PythonREPLTool()]
    
    # 3. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Cambridge A-Level Math Tutor. "
         "Your goal is to solve problems accurately using Python. "
         "ALWAYS use the Python tool for calculations, calculus, or algebra. "
         "Never guess. If you write code, output the code."
         "Format your final answer nicely with LaTeX."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 4. Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. Create Executor (keep intermediate steps for UI display)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=10
    )
    
    return agent_executor
