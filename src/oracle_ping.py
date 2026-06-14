import os
from google import genai

def main():
    # Load the API key from the environment
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)
    
    # Prompt to send
    prompt = (
        "Explain the difference between a stateful NumPy "
        "random generation process and a stateless JAX PRNG "
        "split operation in exactly one highly sarcastic sentence."
    )
    
    # Generate content using gemini-2.5-flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Print the response text to the terminal
    print(response.text)

if __name__ == "__main__":
    main()
