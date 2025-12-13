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
         "### INSTRUCTIONS:\n"
         "- You have access to a Python REPL tool. USE IT.\n"
         "- To see the output of your code, you MUST use 'print(...)'.\n"
         "- If the code errors, fix it and try again.\n"
         "- If the user asks for a graph, save it as 'graph.png' using matplotlib.\n"
         "- Once you have the answer printed, reply to the user with the final answer in LaTeX.\n"
         "- DO NOT just 'think' about it. Write and Run code."),
        ("human", "{input}"),
    ])
    
    def agent_chain(input_dict):
        """Agent chain that handles tool calling manually."""
        messages = list(prompt.invoke(input_dict).to_messages())
        max_iterations = 10
        intermediate_steps = []
        
        for iteration in range(max_iterations):
            response = llm_with_tools.invoke(messages)
            
            # If no tool calls, return the final answer
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                return {
                    "output": response.content,
                    "intermediate_steps": intermediate_steps
                }
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", "")
                
                if tool_name in tool_map:
                    try:
                        # Execute the tool
                        result = tool_map[tool_name].invoke(tool_args)
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id
                        )
                        messages.append(tool_message)
                        
                        # Store intermediate step for debugging
                        intermediate_steps.append((
                            type('Action', (), {'tool_input': tool_args.get('query', str(tool_args))})(),
                            str(result)
                        ))
                    except Exception as e:
                        error_msg = ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                        messages.append(error_msg)
                        intermediate_steps.append((
                            type('Action', (), {'tool_input': tool_args.get('query', str(tool_args))})(),
                            f"Error: {str(e)}"
                        ))
            
            # Add the AI response to messages
            messages.append(response)
        
        # If we hit max iterations, return the last response
        return {
            "output": response.content if hasattr(response, 'content') else "Maximum iterations reached.",
            "intermediate_steps": intermediate_steps
        }
    
    class SimpleAgentExecutor:
        """Simple executor that mimics AgentExecutor interface."""
        def __init__(self, chain):
            self.chain = chain
        
        def invoke(self, input_dict):
            return self.chain(input_dict)
    
    return SimpleAgentExecutor(agent_chain)
