from agent import get_math_agent

def main():
    print("--- A-Level Math Solver ---")
    
    # Initialize the agent once
    agent = get_math_agent()
    
    while True:
        # Get user input
        question = input("\nEnter a math problem (or 'q' to quit): ")
        
        # Check for exit condition
        if question.lower() in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
            
        print(f"\nThinking...")
        try:
            # Pass through directly; prompt already enforces Python & clarity.
            result = agent.invoke({"input": question})
            print(f"\nAnswer: {result['output']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

