import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import sandbox_env

class ControlDecision(BaseModel):
    system_state: str = Field(description="The current state of the system: FREEZING, BOILING, or PERFECT")
    adjustment_action: str = Field(description="The adjustment action to take: INCREASE, DECREASE, or HOLD")
    delta_value: float = Field(description="The change in kappa value to apply")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")

def main():
    # Load API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    
    # Create the client
    client = genai.Client(api_key=api_key)
    
    kappa = 0.1
    
    for turn in range(1, 6):
        # Get current state and logs from sandbox_env
        state, log_msg = sandbox_env.get_system_state(kappa)
        
        # Ask Gemini to return a ControlDecision JSON
        prompt = (
            f"System State: {state}\n"
            f"Telemetry: {log_msg}\n"
            f"Target: Reach PERFECT range (0.4 <= kappa <= 0.8). Currently kappa is {kappa:.4f}.\n"
            f"Decide whether to INCREASE, DECREASE, or HOLD the kappa value, and specify the delta_value."
        )
        
        # Call Gemini with retry logic for high-demand spikes and rate limits
        import time
        max_retries = 6
        response = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ControlDecision,
                        temperature=0.0,
                    )
                )
                break
            except Exception as e:
                err_str = str(e)
                if any(x in err_str for x in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    if attempt == max_retries - 1:
                        raise e
                    # If rate limited, sleep for 65 seconds; otherwise exponential backoff
                    sleep_time = 65 if any(x in err_str for x in ["429", "RESOURCE_EXHAUSTED"]) else (2 ** (attempt + 1))
                    print(f"[Warning] API issue encountered ({err_str[:80]}...). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise e
        
        # Pacing sleep of 12 seconds to avoid hitting the 5 RPM rate limit
        time.sleep(12)
        
        # Parse decision using Pydantic via .parsed
        decision: ControlDecision = response.parsed
        
        action = decision.adjustment_action.upper()
        delta = decision.delta_value
        
        # Update kappa
        if action == "INCREASE":
            kappa += delta
        elif action == "DECREASE":
            kappa -= delta
        # If HOLD, no change to kappa
        
        # Print iteration details as requested
        print(f"Turn {turn} | State: {state} | Action: {decision.adjustment_action} | Delta: {decision.delta_value} | New Kappa: {kappa}")

if __name__ == "__main__":
    main()
