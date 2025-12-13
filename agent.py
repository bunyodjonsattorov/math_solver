import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import ToolMessage, HumanMessage

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
         "- The LAST line of your code MUST be print() with the final answer. Example: print(final_answer)\n"
         "- NEVER just write a variable name like 'final_curve'. You MUST use print(final_curve) to see the output.\n"
         "- If you see output that is just a variable name (like 'final_curve'), it means you forgot to use print(). Add print() immediately.\n"
         "- If the code errors, fix it and try again.\n"
         "- For graphs, save to 'graph.png' using matplotlib, then print confirmation.\n"
         "- After you see the print() output with the actual result, provide the final answer in LaTeX format.\n"
         "- If you've already run similar code multiple times, check if you're missing print() on the last line.\n"
         "- NEVER try to solve problems without using the tool. ALWAYS write and execute code first."),
        ("human", "{input}"),
    ])
    
    def agent_chain(input_dict):
        """Agent chain that handles tool calling manually."""
        messages = list(prompt.invoke(input_dict).to_messages())
        max_iterations = 10
        intermediate_steps = []
        previous_codes = []  # Track previous code to detect loops
        
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
            
            # CRITICAL: Add the AI response (with tool_calls) to messages FIRST
            # OpenAI requires AIMessage with tool_calls to come before ToolMessages
            messages.append(response)
            
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
                        
                        code_input = tool_args.get('query', str(tool_args))
                        
                        # Detect loops - if same code was run recently
                        is_loop = code_input in previous_codes[-2:]  # Check last 2 codes
                        previous_codes.append(code_input)
                        if len(previous_codes) > 5:
                            previous_codes.pop(0)  # Keep only last 5
                        
                        # Check if code doesn't have print() on last line
                        lines = code_input.strip().split('\n')
                        last_line = lines[-1].strip() if lines else ""
                        missing_print = 'print(' not in code_input and iteration > 1
                        is_variable_line = (last_line and not last_line.startswith('#') 
                                          and '=' not in last_line 
                                          and '(' not in last_line 
                                          and (last_line.replace('_', '').replace('.', '').isalnum() or '_' in last_line))
                        
                        # Execute the tool
                        result = tool_map[tool_name].invoke(tool_args)
                        result_str = str(result)
                        
                        # Add helpful feedback if we detect issues
                        if is_loop or (missing_print and is_variable_line):
                            feedback = "\n\n⚠️ HINT: The output above might not show the actual value. "
                            if is_variable_line:
                                feedback += f"Your last line '{last_line}' needs print() to show output. Change it to: print({last_line})"
                            elif is_loop:
                                feedback += "You've run similar code before. Make sure the last line uses print() to display the result."
                            result_str = result_str + feedback
                        
                        tool_message = ToolMessage(
                            content=result_str,
                            tool_call_id=tool_call_id
                        )
                        # Add ToolMessage AFTER the AIMessage
                        messages.append(tool_message)
                        
                        # Store intermediate step for debugging
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
        
        # If we hit max iterations, provide helpful feedback
        last_code = previous_codes[-1] if previous_codes else ""
        if last_code and 'print(' not in last_code:
            lines = last_code.strip().split('\n')
            last_line = lines[-1].strip() if lines else ""
            if last_line and not last_line.startswith('#') and '=' not in last_line:
                hint = f"\n\n💡 TIP: The code was missing print() on the last line. Try adding: print({last_line})"
                final_output = f"Maximum iterations reached. The code needs print() to display results.{hint}"
        else:
            final_output = "Maximum iterations reached. Please check if the code is using print() to display results."
        
        return {
            "output": response.content if hasattr(response, 'content') and response.content else final_output,
            "intermediate_steps": intermediate_steps
        }
    
    class SimpleAgentExecutor:
        """Simple executor that mimics AgentExecutor interface."""
        def __init__(self, chain):
            self.chain = chain
        
        def invoke(self, input_dict):
            return self.chain(input_dict)
    
    return SimpleAgentExecutor(agent_chain)
