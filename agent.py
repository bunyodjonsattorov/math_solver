import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import ToolMessage

# This version uses LLM with tools bound directly - no agent creation functions needed
# This is the most compatible approach that works with any LangChain version

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
    """Build a math agent using LLM with tools bound directly."""
    OPENAI_API_KEY = _get_api_key()
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    tools = [PythonREPLTool()]
    tool_map = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert Cambridge A-Level Math Tutor. "
         "Your goal is to solve the user's problem using Python code. "
         "\n\n"
         "### CRITICAL INSTRUCTIONS:\n"
         "- You MUST use the Python REPL tool to solve EVERY problem. Do not try to solve it without running code.\n"
         "- Write Python code using sympy, numpy, or matplotlib as needed.\n"
         "- The LAST line of your code MUST be print() with the final answer.\n"
         "- To see output, you MUST use 'print(...)'. Never just write a variable name.\n"
         "- If the code errors, fix it and try again.\n"
         "- For graphs, save to 'graph.png' using matplotlib, then print confirmation.\n"
         "- After you see the print() output, provide the final answer in LaTeX format.\n"
         "- NEVER try to solve problems without using the tool. ALWAYS write and execute code first."),
        ("human", "{input}"),
    ])
    
    def agent_chain(input_dict):
        """Agent chain that handles tool calling manually."""
        messages = list(prompt.invoke(input_dict).to_messages())
        max_iterations = 10
        intermediate_steps = []
        
        for iteration in range(max_iterations):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                return {
                    "output": f"Error calling LLM: {str(e)}",
                    "intermediate_steps": intermediate_steps
                }
            
            # Check for tool calls - handle different response formats
            tool_calls = []
            if hasattr(response, 'tool_calls'):
                # tool_calls might be None, empty list, or a list
                if response.tool_calls:
                    tool_calls = response.tool_calls if isinstance(response.tool_calls, list) else [response.tool_calls]
            
            # If no tool calls, return the final answer
            if not tool_calls:
                final_output = response.content if hasattr(response, 'content') else str(response)
                return {
                    "output": final_output or "No response generated.",
                    "intermediate_steps": intermediate_steps
                }
            
            # Execute tool calls
            for tool_call in tool_calls:
                # Handle different tool_call formats
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name", "")
                    tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
                    tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("id", "")
                else:
                    # If it's an object with attributes
                    tool_name = getattr(tool_call, "name", "")
                    tool_args = getattr(tool_call, "args", {})
                    tool_call_id = getattr(tool_call, "id", "")
                
                # If tool_args is a string (JSON), parse it
                if isinstance(tool_args, str):
                    import json
                    try:
                        tool_args = json.loads(tool_args)
                    except:
                        tool_args = {"query": tool_args}
                
                if tool_name in tool_map:
                    try:
                        # PythonREPLTool expects the code as 'query' parameter
                        # If args is empty or doesn't have 'query', use the whole args dict
                        if not tool_args or 'query' not in tool_args:
                            # Try to extract code from args
                            code = tool_args.get('code', tool_args.get('input', str(tool_args)))
                            tool_args = {"query": code}
                        
                        # Execute the tool
                        result = tool_map[tool_name].invoke(tool_args)
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id
                        )
                        messages.append(tool_message)
                        
                        # Store intermediate step for debugging
                        code_input = tool_args.get('query', str(tool_args))
                        intermediate_steps.append((
                            type('Action', (), {'tool_input': code_input})(),
                            str(result)
                        ))
                    except Exception as e:
                        import traceback
                        error_str = f"Error: {str(e)}\n{traceback.format_exc()}"
                        error_msg = ToolMessage(
                            content=error_str,
                            tool_call_id=tool_call_id
                        )
                        messages.append(error_msg)
                        code_input = tool_args.get('query', str(tool_args))
                        intermediate_steps.append((
                            type('Action', (), {'tool_input': code_input})(),
                            error_str
                        ))
            
            # Add the AI response to messages
            messages.append(response)
        
        # If we hit max iterations, return the last response
        return {
            "output": response.content if hasattr(response, 'content') and response.content else "Maximum iterations reached.",
            "intermediate_steps": intermediate_steps
        }
    
    class SimpleAgentExecutor:
        """Simple executor that mimics AgentExecutor interface."""
        def __init__(self, chain):
            self.chain = chain
        
        def invoke(self, input_dict):
            return self.chain(input_dict)
    
    return SimpleAgentExecutor(agent_chain)
