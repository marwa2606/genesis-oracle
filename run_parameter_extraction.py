import json
import asyncio
from google.adk.runners import InMemoryRunner
from scholar_prime.agent import root_agent, search_arxiv, extract_parameters_from_text

def parse_multi_json(text: str) -> dict:
    """
    Parses a string containing multiple consecutive JSON objects and returns the last valid one.
    """
    decoder = json.JSONDecoder()
    pos = 0
    text = text.strip()
    last_obj = {}
    while pos < len(text):
        start_pos = text.find('{', pos)
        if start_pos == -1:
            break
        try:
            obj, idx = decoder.raw_decode(text[start_pos:])
            last_obj = obj
            pos = start_pos + idx
        except json.JSONDecodeError:
            pos = start_pos + 1
    return last_obj

async def main():
    # Initialize the scholar_prime agent
    runner = InMemoryRunner(agent=root_agent)
    print("Scholar-Prime agent initialized.")
    
    # Search the literature using the existing search_arxiv() tool
    query = "thermodynamic simulation parameters for advanced fission reactors"
    print(f"Searching literature for: '{query}'")
    search_results_str = search_arxiv(query)
    
    # Parse results using the robust multi-JSON parser
    search_data = parse_multi_json(search_results_str)
    if not search_data:
        print("Failed to parse search results.")
        print(f"Raw output: {search_results_str}")
        return
        
    papers = search_data.get("papers", [])
    if not papers:
        print("No papers found matching the query in search results.")
        return
        
    # Select the most relevant paper (the first one)
    most_relevant = papers[0]
    print(f"\nMost relevant paper identified: {most_relevant.get('title')}")
    
    # Extract the paper abstract
    abstract = most_relevant.get("summary", "")
    print(f"Abstract preview: {abstract[:200]}...")
    
    # Pass the abstract to extract_parameters_from_text()
    print("\nExtracting parameters from abstract...")
    parameters = extract_parameters_from_text(abstract)
    
    # Fill in the DOI/ID from metadata if it wasn't extracted from the abstract text
    if not parameters.get("doi"):
        if most_relevant.get("doi"):
            parameters["doi"] = most_relevant.get("doi")
        elif most_relevant.get("id"):
            parameters["doi"] = most_relevant.get("id")
        
    print(f"Extracted parameters: {parameters}")
    
    # Save the resulting dictionary as simulation_parameters.json
    output_file = "simulation_parameters.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parameters, f, indent=4)
    print(f"\nSaved parameters to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
