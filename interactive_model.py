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
    
    "theory_threshold": Slider(
        value=0,
        label="Theory change threshold",
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
    ax.set_xlabel("Round")
    ax.set_ylabel("Expected success rate")
    ax.set_title("Agent beliefs vs. true probabilities")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()

    
    solara.FigureMatplotlib(fig)



@solara.component
def agent_stats(model):
    update_counter.get()
    ratio_a = model.Count_State_a() * 100
    ratio_b = model.Count_State_b() * 100
    round_display = model.consensus_round if model.consensus_round else model.round_counter

    update_counter.get()
    conv = model.convergence_status

    config = {
        0: ("#757575", "⏳  No consensus reached"),
        1: ("#2D6A4F", "✅  Consensus on correct hypothesis (A)"),
        2: ("#C0392B", "❌  Consensus on wrong hypothesis (B)"),
    }
    color, text = config[conv]

    solara.HTML(
        tag="div",
        unsafe_innerHTML=text,
        style={
            "background-color": color,
            "color": "white",
            "padding": "10px 20px",
            "border-radius": "5px",
            "font-weight": "bold",
            "font-size": "1rem",
            "display": "inline-block",
            "margin": "10px auto",
            "letter-spacing": "0.3px",
            "transition": "background-color 0.3s ease"
        }
    )

    with solara.Column(style={"gap": "4px"}):
        solara.Text(f"Pursuing A: {ratio_a:.1f}%  |  Pursuing B: {ratio_b:.1f}%")
        solara.Text(f"Round: {round_display}", style={"color": "grey", "fontSize": "0.9rem"})

@solara.component
def experiment_stats(model):
    update_counter.get()

    ra = model.experiments_round_results_a
    rb = model.experiments_round_results_b
    ta = model.experiments_results_a
    tb = model.experiments_results_b

    sra = ra["successes"] / ra["trials"] if ra["trials"] > 0 else 0
    srb = rb["successes"] / rb["trials"] if rb["trials"] > 0 else 0
    eva = ta["successes"] / ta["trials"] if ta["trials"] > 0 else 0
    evb = tb["successes"] / tb["trials"] if tb["trials"] > 0 else 0

    html = f"""
    <div style="font-family: sans-serif; font-size: 0.9rem; margin: 8px 0;">
      <div style="font-weight: 600; font-size: 1rem; margin-bottom: 10px;">
        Experimental Evidence
      </div>
      <table style="border-collapse: collapse; width: 100%;">
        <thead>
          <tr>
            <th style="text-align:left; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;"></th>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;
                       color: #2D6A4F;">Theory A</th>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;
                       color: #C0392B;">Theory B</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding: 6px 12px; color: #888; font-size: 0.82rem;">This round</td>
            <td style="text-align:center; padding: 6px 12px;">
              {ra['successes']} / {ra['trials']}
              <span style="color:#aaa; font-size:0.82rem;"> ({sra:.3f})</span>
            </td>
            <td style="text-align:center; padding: 6px 12px;">
              {rb['successes']} / {rb['trials']}
              <span style="color:#aaa; font-size:0.82rem;"> ({srb:.3f})</span>
            </td>
          </tr>
          <tr style="background: rgba(0,0,0,0.03);">
            <td style="padding: 6px 12px; color: #888; font-size: 0.82rem;">Cumulative</td>
            <td style="text-align:center; padding: 6px 12px;">
              {ta['successes']} / {ta['trials']}
              <span style="color:#aaa; font-size:0.82rem;"> ({eva:.3f})</span>
            </td>
            <td style="text-align:center; padding: 6px 12px;">
              {tb['successes']} / {tb['trials']}
              <span style="color:#aaa; font-size:0.82rem;"> ({evb:.3f})</span>
            </td>
          </tr>
          <tr style="border-top: 1px solid #e0e0e0;">
            <td style="padding: 8px 12px; color: #888; font-size: 0.82rem;">Empirical p̂</td>
            <td style="text-align:center; padding: 8px 12px;
                       font-weight: 700; color: #2D6A4F; font-size: 1rem;">{eva:.4f}</td>
            <td style="text-align:center; padding: 8px 12px;
                       font-weight: 700; color: #C0392B; font-size: 1rem;">{evb:.4f}</td>
          </tr>
        </tbody>
      </table>
    </div>
    """

    solara.HTML(tag="div", unsafe_innerHTML=html)


epistemic_ABM = Bandit()


SpaceViz = make_space_component(
    agent_portrayal, backend="matplotlib", layout_alg = nx.shell_layout, layout_kwargs={"scale": 1.0}
)

@solara.component
def Page():
    solara.Title("Epistemic Bandit Model")
    solara.lab.theme.themes.light.primary = "#2D6A4F"
    solara.lab.theme.themes.dark.primary  = "#2D6A4F" 
    solara.Style(".v-tab { display: none !important; }")     

    viz = SolaraViz(
        epistemic_ABM,
        model_params=model_params,
        components=[SpaceViz, deviation_plot, agent_stats, experiment_stats],
        name="Epistemic Bandit ABM",
    )