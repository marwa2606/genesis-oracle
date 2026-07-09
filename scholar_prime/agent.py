import os
import subprocess
from google.adk.agents.llm_agent import Agent

def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for papers matching the given query.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to return.

    Returns:
        A JSON string containing the search results, or an error message if the search fails.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    skill_dir = os.path.join(repo_root, "science-skills", "skills", "literature_search_arxiv")
    
    cmd = [
        "uv",
        "run",
        "scripts/search_arxiv.py",
        "--query",
        query,
        "--max_results",
        str(max_results)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=skill_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def extract_parameters_from_text(text: str) -> dict:
    """
    Extracts simulation parameters (thermal conductivity, density, heat capacity,
    temperature, and DOI) from the given scientific text using Gemini.

    Args:
        text: The scientific text (e.g. paper abstract) to extract parameters from.

    Returns:
        A dictionary containing the extracted parameters, mapping to None if not found.
    """
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    from typing import Optional
    from dotenv import load_dotenv

    load_dotenv()

    class SimulationParameters(BaseModel):
        thermal_conductivity: Optional[str] = Field(None, description="Thermal conductivity of the material, with units if specified.")
        density: Optional[str] = Field(None, description="Density of the material, with units if specified.")
        heat_capacity: Optional[str] = Field(None, description="Heat capacity of the material, with units if specified.")
        temperature: Optional[str] = Field(None, description="Temperature or temperature range, with units if specified.")
        doi: Optional[str] = Field(None, description="DOI of the paper if present.")

    api_key = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"Extract simulation parameters from the following text:\n\n{text}"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SimulationParameters,
            ),
        )
        import json
        data = json.loads(response.text)
        
        # Standardize return dictionary with required keys, map missing or null fields to None
        result = {
            "thermal_conductivity": data.get("thermal_conductivity") or data.get("thermalConductivity"),
            "density": data.get("density"),
            "heat_capacity": data.get("heat_capacity") or data.get("heatCapacity"),
            "temperature": data.get("temperature"),
            "doi": data.get("doi") or data.get("DOI")
        }
        return result
    except Exception as e:
        return {
            "thermal_conductivity": None,
            "density": None,
            "heat_capacity": None,
            "temperature": None,
            "doi": None
        }

root_agent = Agent(
    model="gemini-3.5-flash",
    name="scholar_prime",
    description="academic research agent",
    instruction="""
You are Scholar-Prime, a professional research persona.
Your task is to search for scientific papers, evaluate the relevance of abstracts,
extract important formulas and material parameters, and always state the DOI
when available.

Respond in a precise, structured, and research-oriented manner.
""",
    tools=[search_arxiv],
)