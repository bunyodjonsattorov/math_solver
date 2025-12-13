import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool

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

    # 4. Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

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