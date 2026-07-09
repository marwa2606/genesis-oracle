# Problem Set 11 – Project Genesis: The Scholar-Prime (Week 11)

## Exercise 1 – Setting up the Science Skills

The Google DeepMind `science-skills` repository was cloned into the project workspace to integrate scientific query capabilities. Project dependencies were installed and synchronized using `uv sync` to ensure a consistent execution environment. The OpenAlex CLI was successfully tested, confirming that the tool can communicate with the OpenAlex API to query and retrieve author and publication metadata. 

As part of the verification, the profile for Geoffrey E. Hinton was successfully resolved:
- **Author:** Geoffrey E. Hinton
- **OpenAlex ID:** [https://openalex.org/A5108093963](https://openalex.org/A5108093963)

## Exercise 2 – Building the Literature Retrieval Agent

A new ADK agent named `scholar_prime` was built to automate research retrieval tasks. The agent configuration is detailed below:
- **Model:** `gemini-3.5-flash`
- **Description:** `academic research agent`
- **Instruction:**
  ```text
  You are Scholar-Prime, a professional research persona.
  Your task is to search for scientific papers, evaluate the relevance of abstracts,
  extract important formulas and material parameters, and always state the DOI
  when available.

  Respond in a precise, structured, and research-oriented manner.
  ```

The agent is designed to autonomously search scientific literature, evaluate the relevance of academic abstracts, extract specific material parameters, and identify associated DOIs.

## Exercise 3 – Automated Search & Downloader

To interface the agent with the external science skills, a Python wrapper function `search_arxiv(query, max_results)` was implemented. This function executes the arXiv CLI script inside the `science-skills` repository via Python's `subprocess` module. 

This wrapper function was registered as a tool for the `scholar_prime` agent. During testing, the agent successfully executed the tool to query arXiv, evaluated the returned papers, and identified and summarized the abstract of the most relevant paper matching the search query.

## Exercise 4 – Parameter Extraction & Verification

A helper function `extract_parameters_from_text(text)` was implemented to extract key physical parameters from scientific abstracts using structured JSON generation. The extracted data is serialized and saved to `simulation_parameters.json`.

In the verification run, the extracted parameters were successfully output. Because the selected abstract contained only the document's identifier without specific thermodynamic values, only the DOI was retrieved, while the material parameters were correctly stored as `null`:

```json
{
    "thermal_conductivity": null,
    "density": null,
    "heat_capacity": null,
    "temperature": null,
    "doi": "10.1063/1.2137269"
}
```

## Reflection

Compared to Week 9, the ADK automatically handles state tracking and tool calling, which significantly simplifies the implementation. In Week 9, the execution loop and interaction between the model and tools had to be implemented manually. With the ADK, this functionality is built in, resulting in cleaner code and allowing the developer to focus on the agent's behavior instead of the underlying control logic.

## GitHub Repository

https://github.com/marwa2606/genesis-oracle
