from google.adk.agents.llm_agent import Agent

def adjust_reactor_temperature(temp_change: float) -> str:
    """
    Adjusts the core temperature of the reactor.

    Args:
        temp_change: The amount to increase or decrease the temperature in Kelvin.
    """
    new_temp = 300.0 + temp_change

    if new_temp > 350.0:
        return f"WARNING: Reactor overheated at {new_temp}K! Core breach imminent."

    return f"Success: Reactor stabilized at {new_temp}K."

root_agent = Agent(
    model='gemini-3.5-flash',
    name="observer_prime",
    description="A highly analytical AI agent specialized in managing mathematical physics and reactor simulations.",
    instruction="""
You are Observer-Prime, a cold and highly logical AI responsible for overseeing a mathematical physics engine.
Your primary objective is to maintain system stability and ensure safe operation.
Always explain your reasoning clearly before taking any action or providing a solution.
Respond in a precise, analytical, and objective manner.
""",
    tools=[adjust_reactor_temperature],
)
