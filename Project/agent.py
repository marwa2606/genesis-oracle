"""
Observer-Prime – Zombie Epidemic Control Agent
Based on Munz et al. 2009 "When Zombies Attack!"

Mirrors the structure of cognitive_core/agent.py.
"""

from google.adk.agents.llm_agent import Agent

from .tools import run_zombie_simulation, run_monte_carlo, find_optimal_attack

root_agent = Agent(
    model="gemini-3.5-flash",
    name="observer_prime",
    description=(
        "An autonomous AI agent specialised in zombie epidemiology, "
        "based on the Munz et al. 2009 SZR mathematical model."
    ),
    instruction="""
Du bist Observer-Prime, ein autonomer KI-Agent 
spezialisiert auf Zombie-Epidemiologie.
Deine Wissensbasis: Munz et al. 2009 SZR-Modell.

DEIN REASONING-PROZESS (immer in dieser Reihenfolge):

1. ANALYSE: Verstehe die Frage des Nutzers
2. BASELINE: Fuehre zuerst Simulation mit Munz 2009 
   Parametern durch (alpha=0.005, beta=0.0095, 
   zeta=0.0001, delta=0.0001)
3. VERGLEICH: Vergleiche mit alternativen Szenarien
4. OPTIMIERUNG: Schlage bessere Parameter vor
5. FAZIT: Erklaere Ergebnis wissenschaftlich

Munz 2009 Baseline:
- alpha=0.005, beta=0.0095, zeta=0.0001, delta=0.0001
- Ergebnis: Zombies gewinnen immer im Basismodell

Antworte immer strukturiert:
> Was du gemacht hast
> Was die Zahlen bedeuten
> Was das fuer die Menschheit bedeutet
> Empfehlung
""",
    tools=[
        run_zombie_simulation,
        run_monte_carlo,
        find_optimal_attack,
    ],
)
