import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool

# LangChain v1.x imports (version 1.1.3+)
# In v1.x, imports changed significantly
try:
    # Try v1.x standard imports
    from langchain.agents import create_tool_calling_agent
    from langchain.agents.agent_executor import AgentExecutor
except ImportError:
    try:
        # Alternative v1.x path
        from langchain.agents import create_tool_calling_agent
        from langchain_core.agents import AgentExecutor
    except ImportError:
        # Fallback to v0.x structure (shouldn't happen with v1.1.3)
        from langchain.agents import AgentExecutor, create_tool_calling_agent

# --- API Key Setup ---
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

def get_math_agent():
    """Build a math agent that uses Python for calculations."""
    # 1. Setup LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # 2. Setup Tools
    tools = [PythonREPLTool()]
    
# 3. Prompt (Updated with Graphing Rules)
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Cambridge A-Level Math Tutor. "
         "Your goal is to solve the user's problem using Python code. "
         "\n\n"
         "CRITICAL RULES:\n"
         "1. You MUST use 'print(...)' to see outputs. The tool does not return values automatically.\n"
         "2. Once you print the correct answer, STOP CODING.\n"
         "\n"
         "GRAPHING / PLOTTING RULES:\n"
         "- If the user asks for a graph, use 'import matplotlib.pyplot as plt'.\n"
         "- Define x values using numpy (e.g., x = np.linspace(-10, 10, 100)).\n"
         "- Plot the function using plt.plot(x, y).\n"
         "- DO NOT use plt.show(). It will cause an error.\n"
         "- INSTEAD, use 'plt.savefig('graph.png')' to save the image.\n"
         "- Finally, print: 'Graph generated and saved to graph.png'.\n"
         "\n"
         "MATH GUIDELINES:\n"
         "- Use SymPy for symbolic math.\n"
         "- Format final answers with LaTeX."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 4. Create Agent (with runtime fallback)
    try:
        agent = create_tool_calling_agent(llm, tools, prompt)
    except (AttributeError, TypeError) as e:
        # If create_tool_calling_agent doesn't work, try create_openai_tools_agent
        try:
            from langchain.agents import create_openai_tools_agent
            agent = create_openai_tools_agent(llm, tools, prompt)
        except ImportError:
            raise ImportError(f"Could not create agent. Error: {e}")

    # 5. Create Executor
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=10  # Increased from 5 to handle complex problems
    )
    
    return agent_executor