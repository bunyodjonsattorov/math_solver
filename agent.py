import os
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

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
    
    # Strict prompt that forces Python execution and printing
    strict_prefix = """You are a math solver agent. You MUST write and execute Python code to solve every problem.

CRITICAL RULES:
1. ALWAYS write Python code using sympy, numpy, or matplotlib. NEVER guess or answer without running code.
2. The LAST line of your code MUST be print() with the final answer. Do NOT just write a variable name - you MUST use print().
3. After you see the print() output, STOP. Do NOT run the same code again. The print() output IS your final answer.
4. For integration problems: integrate, substitute any given points to solve for constants (like C), substitute back, then PRINT the final expression using print().
5. For graphing: use matplotlib, save to 'graph.png', then print("Graph saved to graph.png").
6. If you cannot write code to solve it, return "I don't know" - do NOT guess.

CORRECT Example for integration:
```python
import sympy as sp
x, C = sp.symbols('x C')
dy_dx = 12*(2*x - 5)**2 + 8*x
y = sp.integrate(dy_dx, x) + C
C_value = sp.solve(y.subs(x, 2) - 4, C)[0]
y_final = y.subs(C, C_value)
print(y_final)  # THIS IS REQUIRED - last line must be print()
```

WRONG - Do NOT do this:
```python
y_final  # This will NOT work - you must use print()
```

Remember: The last line MUST be print(). After you see the output, STOP. Do NOT repeat the same code."""
    
    # Create Python agent with strict prompt
    tool = PythonREPLTool()
    agent_executor = create_python_agent(
        llm=llm,
        tool=tool,
        verbose=True,
        prefix=strict_prefix,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
            "max_iterations": 15,
            "return_intermediate_steps": True
        }
    )
    
    return agent_executor