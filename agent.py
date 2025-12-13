import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent

def _get_api_key():
    """Get API key from secrets or environment."""
    OPENAI_API_KEY = None
    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.secrets:
            OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Check your secrets or environment variables.")
    
    return OPENAI_API_KEY

def get_math_agent():
    """Build a math agent that uses Python for calculations."""
    # Get API key (only when agent is created, not during import)
    OPENAI_API_KEY = _get_api_key()
    
    # Setup LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Create Python agent (this is more stable across LangChain versions)
    # create_python_agent handles the AgentExecutor setup internally
    agent_executor = create_python_agent(
        llm=llm,
        tool=None,  # Uses PythonREPLTool by default
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=10
    )
    
    return agent_executor