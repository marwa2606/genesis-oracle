import os
from google import genai
from google.genai import types

def main():
    # 1. Load data/audit_target.png as bytes
    image_path = os.path.join("data", "audit_target.png")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}. Please run generate_signals.py first.")
        
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # 2. Load API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # 3. Create genai.Client()
    client = genai.Client(api_key=api_key)

    # 4. Prompt for the multimodal request
    prompt = (
        "You are a Visual Detective. Analyze this signal plot. "
        "Find the visual anomaly/malfunction, guess the exact "
        "X-axis region where it happened, and write a short "
        "funny poem mocking the engineering team that allowed "
        "this bug to pass."
    )

    # 5. Send multimodal request to gemini-2.5-flash
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ),
            prompt
        ]
    )

    # 6. Print Gemini's response
    print(response.text)

if __name__ == "__main__":
    main()
