import os
import ast
import io
import contextlib
import sys
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

# --- Safe Execution Environment ---

class RestrictedPythonTool:
    """
    A custom tool that executes Python code with:
    1. Automatic output capture (no print() needed for last line)
    2. Import restrictions (only math libraries allowed)
    3. Loop detection integrated in the agent
    """
    name = "python_repl"
    description = "Executes Python code. Use this to solve math problems. The last line contributes to the return value."

    def __init__(self):
        self.locals = {}
        # Pre-import common libraries for convenience
        self._exec_code("import math\nimport numpy as np\nimport sympy\nimport matplotlib.pyplot as plt")

    def _exec_code(self, code_str):
        """Executes code and returns the value of the last expression."""
        # capture stdout
        stdout_capture = io.StringIO()
        
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return f"Syntax Error: {e}"

        # Security Check: Prevent dangerous imports in the code node
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    module_name = name.name.split('.')[0]
                    if module_name not in ['math', 'numpy', 'sympy', 'matplotlib', 'pandas', 'scipy', 'fractions', 'decimal']:
                        return f"Security Error: Import of '{module_name}' is restricted. Only math/science libraries are allowed."

        # Separate the last node if it's an expression
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()

        # Execute body (everything except potentially the last expression)
        try:
            with contextlib.redirect_stdout(stdout_capture):
                # We need to compile the modified tree
                if tree.body:
                    exec(compile(tree, filename="<string>", mode="exec"), {}, self.locals)
                
                # Evaluate the last expression if it exists
                if last_expr:
                    result = eval(compile(ast.Expression(last_expr.value), filename="<string>", mode="eval"), {}, self.locals)
                    if result is not None:
                        print(result) # Print it so it goes to stdout
        
        except Exception as e:
            return f"Execution Error: {e}"

        return stdout_capture.getvalue().strip() or "Code executed successfully (no output)"

    def invoke(self, args):
        # Handle different arg formats from LLM
        if isinstance(args, str):
            import json
            try:
                args = json.loads(args)
            except:
                return self._exec_code(args) # Treat raw string as code
        
        code = args.get("query", args.get("code", ""))
        return self._exec_code(code)


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
    """Build a math agent using LLM with custom tools."""
    OPENAI_API_KEY = _get_api_key()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    # Use our custom restricted tool
    repl_tool = RestrictedPythonTool()
    
    # We need to wrap it to be compatible with bind_tools
    @tool("python_repl")
    def execute_python(query: str):
        """Executes Python code. Use this for all math calculations. Returns the output of print() or the last expression."""
        return repl_tool.invoke({"query": query})

    tools = [execute_python]
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Cambridge A-Level Math Tutor. "
         "Your goal is to solve the user's problem using Python code. "
         "\n\n"
         "### INSTRUCTIONS:\n"
         "- ALWAYS use the 'python_repl' tool to solve math problems. Do NOT do math in your head.\n"
         "- Write clean Python code using sympy, numpy, or matplotlib.\n"
         "- The tool automatically captures the result of the last line, so you don't need to force print() everywhere, but print() is still good practice.\n"
         "- Check the tool output. If it says 'Security Error', you are importing a restricted library.\n"
         "- For graphs: save to 'graph.png' using matplotlib.pyplot.\n"
         "- Provide the final answer in clear LaTeX or text after the code execution confirms it.\n"),
        ("human", "{input}"),
    ])
    
    def agent_chain(input_dict):
        """Agent chain that handles tool calling manually."""
        conversation_history = input_dict.get('conversation_history', [])
        
        # Build messages with conversation history
        # We manually construct the input message to properly include history
        system_msg = prompt.invoke({"input": ""}).to_messages()[0]
        
        messages = [system_msg]
        
        # Add conversation history (last 6 messages)
        for msg in conversation_history[-6:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))
        
        # Add current user input
        messages.append(HumanMessage(content=input_dict.get('input', '')))
        
        max_iterations = 10
        intermediate_steps = []
        previous_outputs = []  # Track previous outputs to detect loops
        
        for iteration in range(max_iterations):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                return {
                    "output": f"Error calling LLM: {str(e)}",
                    "intermediate_steps": intermediate_steps
                }
            
            # Check for tool calls
            tool_calls = response.tool_calls if hasattr(response, 'tool_calls') and response.tool_calls else []
            
            # If no tool calls, return the final answer
            if not tool_calls:
                final_output = response.content
                return {
                    "output": final_output or "No response generated.",
                    "intermediate_steps": intermediate_steps
                }
            
            # Add AI response to history
            messages.append(response)
            
            # Execute tool calls
            for tool_call in tool_calls:
                tool_output = execute_python.invoke(tool_call['args'])
                
                # Advanced Loop Detection
                # If we get the exact same output twice in the last 5 steps, stop.
                if tool_output in previous_outputs[-5:]:
                     tool_output += "\n\n(System Warning: You have received this exact output recently. Stop and think: are you in a loop?)"
                
                previous_outputs.append(tool_output)
                if len(previous_outputs) > 5:
                    previous_outputs.pop(0)

                messages.append(ToolMessage(
                    content=tool_output,
                    tool_call_id=tool_call['id']
                ))
                
                intermediate_steps.append((
                    type('Action', (), {'tool_input': str(tool_call['args'].get('query', ''))})(),
                    tool_output
                ))
        
        return {
            "output": "I reached the maximum number of steps without finding a final answer. Please try simplifying the problem.",
            "intermediate_steps": intermediate_steps
        }
    
    class SimpleAgentExecutor:
        def __init__(self, chain):
            self.chain = chain
        def invoke(self, input_dict):
            return self.chain(input_dict)
    
    return SimpleAgentExecutor(agent_chain)
