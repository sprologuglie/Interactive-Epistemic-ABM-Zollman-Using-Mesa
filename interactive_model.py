###                 APP                 ###

import solara
import solara.lab
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_space_component
)
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter
from matplotlib.figure import Figure
import networkx as nx
from scripts.model import Bandit


def agent_portrayal(agent):
    """Portrays agents as green if they pull A and red if they pull B"""
    return AgentPortrayalStyle(color = "green" if agent.state == "a" else "red", size=90)

model_params = {
    
    "graph": { 
        "type": "Select",
        "value": "complete",
        "label": "Type of Graph",
        "values": ["complete", "cycle", "wheel"]
    },

    "n": {
        "type": "SliderInt",
        "value": 10,
        "label": "Number of agents:",
        "min": 4,
        "max": 100,
        "step": 1,
    },

    "a_objective": Slider(
        value= .5,
        label="Objective probability of A",
        min=.5,
        max=1,
        step=.001
    ),

    "b_objective": Slider(
        value= .499,
        label="Objective probability of B",
        min=0,
        max=.499,
        step=.001
    ),

    "max_priors": {
        "type": "Select",
        "value": 4,
        "label": "Max prior beliefs range",
        "values" : [1, 4, 100, 1000, 5000, 10000]
    },

    "step_pulls": {
        "type": "Select",
        "value": 1000,
        "label": "Pulls per step",
        "values" : [1, 10, 100, 1000, 5000, 10000]
    },

    "dynamic": {
        "type": "Select",
        "label": "Dynamic update of objectives (rounds)",
        "value": None,
        "values": [None, 1, 5, 10, 50, 100, 500, 1000]
    },

    "criticism": {
        "type": "Select",
        "label": "Critical interaction",
        "value": False,
        "values": [False, True]

    },
    
    "theory_treshold": Slider(
        value=0,
        label="Theory change treshold",
        min=0,
        max=.2,
        step=.005
    ),

    "inertia": Slider(
        value=0,
        label="Rounds of inertia",
        min=0,
        max=20,
        step=1
    )
    
    
    
}

@solara.component
def deviation_plot(Model):
    """Plot component for avg. A and B expectations and deviance from objective"""
    update_counter.get()
    
    fig = Figure()
    ax = fig.subplots()
    dev = Model.datacollector.get_model_vars_dataframe()
    ax.plot("Avg. A expectation", data=dev, label="Avg. A expectation")
    ax.plot("A objective probability", data=dev, label="A objective probability", linestyle= "dashed")
    ax.plot("Avg. B expectation", data=dev, label="Avg. B expectation")
    ax.plot("B objective probability", data=dev, label="B objective probability", linestyle = "dashed")
    ax.get_label()
    ax.legend()
    fig.align_xlabels()
    

    
    solara.FigureMatplotlib(fig)



def get_perc_a_believers(model):
    """Displays the number of agents pulling A"""
    ratio = model.Count_State_a() * 100
    ratio_text = f"{ratio:.2f}"
    cons = model.Check_Convergence()

    if cons == 0:
        consensus = "not reached"
    elif cons == 1:
        consensus = "on the CORRECT hypothesis reached"
  
    else: consensus = "on the WRONG hypothesis reached"

    if model.consensus_round == None:
        round = model.round_counter
    else: round = model.consensus_round

    return solara.Markdown(
        f"Percentage of agent pursuing theory A: {ratio_text}%<br>Percentage of agent pursuing theory B: {100 - ratio}%<br>Consensus {consensus} on round {round} "
    )

def get_experiments_round_results(model):
    """Function for displayng step and total experimentals results"""
    res_round_a= model.experiments_round_results_a
    res_round_b = model.experiments_round_results_b
    res_tot_a = model.experiments_results_a
    res_tot_b = model.experiments_results_b
    eva = res_tot_a["successes"] / res_tot_a["trials"] if res_tot_a["trials"] > 0 else 0
    evb = res_tot_b["successes"] / res_tot_b["trials"] if res_tot_b["trials"] > 0 else 0
    return solara.Markdown(
        f"STEP:<br>Number of experiments for A: {res_round_a['trials']} with {res_round_a['successes']} successes <br>Number of experiments for B: {res_round_b['trials']} with {res_round_b['successes']} successes<br>TOTAL:<br>Total number of experiments for A: {res_tot_a['trials']} with {res_tot_a['successes']} successes<br>Total number of experiments for B: {res_tot_b['trials']} with {res_tot_b['successes']} successes<br>Evidence based A = {eva}<br>Evidence based B = {evb}"
    ) 


epistemic_ABM = Bandit()


SpaceViz = make_space_component(
    agent_portrayal, backend="matplotlib", layout_alg = nx.shell_layout, layout_kwargs={"scale": 1.0}
)

@solara.component
def Page():
    solara.Title("Epistemic Bandit Model")
    solara.lab.theme.themes.light.primary = "#2D6A4F"
    solara.lab.theme.themes.dark.primary  = "#2D6A4F"       

    viz = SolaraViz(
        epistemic_ABM,
        model_params=model_params,
        components=[SpaceViz, deviation_plot, get_perc_a_believers, get_experiments_round_results],
        name="Epistemic Bandit ABM",
    )