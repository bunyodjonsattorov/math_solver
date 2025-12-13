from agent import get_math_agent

def main():
    print("--- A-Level Math Solver (CLI) ---")
    print("Type 'q' or 'quit' to exit\n")
    
    # Initialize the agent once
    agent = get_math_agent()
    
    # Conversation history for CLI
    conversation_history = []
    
    while True:
        # Get user input
        question = input("\nEnter a math problem (or 'q' to quit): ")
        
        # Check for exit condition
        if question.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
            
        print(f"\n🧠 Thinking...")
        try:
            # Add user message to history
            conversation_history.append({"role": "user", "content": question})
            
            # Invoke agent with conversation history
            result = agent.invoke({
                "input": question,
                "conversation_history": conversation_history[:-1]  # Exclude current message
            })
            
            # Display answer
            print(f"\n📝 Answer:\n{result['output']}")
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": result['output']})
            
            # Show intermediate steps if available
            steps = result.get('intermediate_steps', [])
            if steps:
                print(f"\n💻 Python Code Executed:")
                for i, step in enumerate(steps, 1):
                    try:
                        print(f"\n  Step {i}:")
                        print(f"  Code: {step[0].tool_input}")
                        print(f"  Output: {step[1][:200]}...")  # Truncate long outputs
                    except:
                        pass
                        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

