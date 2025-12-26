###                 APP                 ###
import math

import solara

from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
)
from mesa.visualization.components import AgentPortrayalStyle
from matplotlib.figure import Figure

from mesa.visualization.utils import update_counter    

def agent_portrayal(agent):
    """Portrays agents as green if they pull A and red if they pull B"""
    return AgentPortrayalStyle(color = "green" if agent.state == "a" else "red", size=90)

model_params = {
    "n": {
        "type": "SliderInt",
        "value": 10,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },

    "a_objective": Slider(
        value= .5,
        label="Objective probability of A",
        min=.5,
        max=1,
        step=.01
    ),

    "b_objective": Slider(
        value= .49,
        label="Objective probability of B",
        min=0,
        max=.49,
        step=.01
    ),

    "stubborness_range": Slider(
        value=0,
        label="Max stubborness",
        min=0,
        max=5,
        step=1
    ),

    "max_priors": {
        "type": "SliderInt",
        "value": 4,
        "label": "Max prior belief",
        "min": 2,
        "max": 10,
        "step": 1
    },

    "graph": { 
        "type": "Select",
        "value": "complete",
        "label": "Type of Graph",
        "values": ["complete", "cycle", "wheel"]
    }
}

@solara.component
def deviation_plot(Model):
    """Plot component for avg. A and B expectations and deviance from objective"""
    update_counter.get()
    
    fig = Figure()
    ax = fig.subplots()
    dev = Model.datacollector.get_model_vars_dataframe()
    ax.plot("Avg. deviance from objectve", data=dev, label="Avg. expectation/objectve")
    ax.plot("Avg. A expectation", data=dev, label="Avg. A expectation")
    ax.plot("Avg. B expectation", data=dev, label="Avg. B expectation")
    ax.get_label()
    ax.legend()

    
    solara.FigureMatplotlib(fig)



def get_perc_a_believers(model):
    """Displays the number of agents pulling A"""
    ratio = model.Count_State_a() * 100
    ratio_text = f"{ratio:.2f}"

    return solara.Markdown(
        f"Percentage of agent pursuing theory a: {ratio_text}%"
    )

def get_experiments_round_results(model):
    """Function for displayng step and total experimentals results"""
    res_round_a, res_round_b = model.Count_Evidence()
    res_tot_a = model.experiments_results_a
    res_tot_b = model.experiments_results_b
    eva = res_tot_a["successes"] / res_tot_a["trials"] if res_tot_a["trials"] > 0 else 0
    evb = res_tot_b["successes"] / res_tot_b["trials"] if res_tot_b["trials"] > 0 else 0
    return solara.Markdown(
        f"STEP:<br>Number of experiments for A: {res_round_a["trials"]} with {res_round_a["successes"]} successes <br>Number of experiments for B: {res_round_b["trials"]} with {res_round_b["successes"]} successes<br>TOTAL:<br>Total number of experiments for A: {res_tot_a["trials"]} with {res_tot_a["successes"]} successes<br>Total number of experiments for B: {res_tot_b["trials"]} with {res_tot_b["successes"]} successes<br>Evidence based a = {eva}<br>Evidence based b = {evb}"
    ) 


zollman_model = Bandit()

renderer = SpaceRenderer(model=zollman_model, backend="matplotlib").render(agent_portrayal=agent_portrayal)

@solara.component
def Page():
    viz = SolaraViz(
        zollman_model,
        renderer,
        model_params=model_params,
        components=[deviation_plot, get_perc_a_believers, get_experiments_round_results],
        name="Zollman Model"
    )
    return solara.Column([
        solara.Markdown("## Zollman's Bandit Model"),
        viz
    ])
###!!!! To run the model use the command "solara run run.py" in the terminal