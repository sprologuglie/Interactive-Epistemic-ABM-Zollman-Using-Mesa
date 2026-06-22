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
import numpy as np
import io
import json
import zipfile
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
    ),

    "seed": {
        "type": "Select",
        "value": None,
        "label": "Seed",
        "values" : [None, 3, 42, 123, 456, 789, 1000, 2839, 8182, 9999]
    }
    
    
    
    
}

@solara.component
def deviation_plot(Model):
    """Plot component for avg. A and B expectations and deviance from objective"""
    update_counter.get()
    
    
    fig = Figure()
    ax = fig.subplots()
    try:
        dev = Model.datacollector.get_model_vars_dataframe()
        if len(dev) == 0:
            solara.FigureMatplotlib(fig)
            return
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

    except (RuntimeError, ValueError):
        pass
    
    solara.FigureMatplotlib(fig)



@solara.component
def agent_stats(model):
    update_counter.get()
    ratio_a = model.count_state_a() * 100
    ratio_b = model.count_state_b() * 100
    round_display = model.consensus_round if model.consensus_round else "no consensus"
    round_counter = model.round_counter

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
        solara.Text(f"Round of current consensus: {round_display}")
        solara.Text(f"Current round: {round_counter}", style={"color": "grey", "fontSize": "0.9rem"})

        solara.FileDownload(
            lambda: generate_results_zip(model),
            filename="epistemic_bandit_results.zip",
            label="Download Results zip",
            mime_type="application/zip"
        )

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

@solara.component
def agents_table(model):
    update_counter.get()

    rows = ""
    for agent in model.agents:
        rows += f"""<tr>
        <td  style="text-align:center; padding: 6px 12px;">{agent.unique_id}</td>
        <td  style="text-align:center; padding: 6px 12px;">{agent.state.upper()}</td>
        <td  style="text-align:center; padding: 6px 12px;">{agent.a_expectations():.3f}</td>
        <td  style="text-align:center; padding: 6px 12px;">{agent.b_expectations():.3f}</td>
        <td  style="text-align:center; padding: 6px 12px;">{agent.a_objective:.3f}</td>
        <td  style="text-align:center; padding: 6px 12px;">{agent.b_objective:.3f}</td>
        </tr>"""

    html = f"""
    <div style="font-family: sans-serif; font-size: 0.9rem; margin: 8px 0; max-height; 500px; overflow-y: auto;">
      <div style="font-weight: 600; font-size: 1rem; margin-bottom: 10px;">
        Agents
      </div>
    <table style="width: 100%;">
    <thead>
        <tr>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;">ID</th>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;">Pursued Theory</th>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;
                       color: #2D6A4F;">Theory A Expectation</th>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;
                       color: #C0392B;">Theory B Expectation</th>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;
                       color: #2D6A4F;">Theory A Objective</th>
            <th style="text-align:center; padding: 6px 12px; border-bottom: 2px solid #e0e0e0;
                       color: #C0392B;">Theory B Objective</th>
        </tr>
    </thead>
    <tbody>{rows}</tbody>
    </table>
    </div>
    """

    solara.HTML(tag="div", unsafe_innerHTML=html)

@solara.component
def aggregate_metrics(model):
    update_counter.get()

    # Calcolo varianze
    beliefs_a = [agent.a_expectations() for agent in model.agents]
    beliefs_b = [agent.b_expectations() for agent in model.agents]
    variance_a = np.var(beliefs_a)
    variance_b = np.var(beliefs_b)

    # Calcolo entropia di Shannon sulla distribuzione A/B
    p_a = model.count_state_a()
    p_b = model.count_state_b()
    entropy = 0
    if p_a > 0:
        entropy -= p_a * np.log2(p_a)
    if p_b > 0:
        entropy -= p_b * np.log2(p_b)

    html = f"""
    <div style="display: flex; gap: 12px; margin: 8px 0;">
        <div style="flex: 1; padding: 12px 16px; border-radius: 8px;
                    background: #f5f5f5; text-align: center;">
            <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">
                Variance — Theory A</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #2D6A4F;">
                {variance_a:.5f}</div>
        </div>
        <div style="flex: 1; padding: 12px 16px; border-radius: 8px;
                    background: #f5f5f5; text-align: center;">
            <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">
                Variance — Theory B</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #C0392B;">
                {variance_b:.5f}</div>
        </div>
        <div style="flex: 1; padding: 12px 16px; border-radius: 8px;
                    background: #f5f5f5; text-align: center;">
            <div style="font-size: 0.8rem; color: #888; margin-bottom: 4px;">
                Shannon Entropy — A/B states</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #555;">
                {entropy:.3f} <span style="font-size: 0.9rem; color: #aaa;">/ 1</span>
            </div>
        </div>
    </div>
    """

    solara.HTML(tag="div", unsafe_innerHTML=html)


@solara.component
def belief_histogram(model):
    update_counter.get()

    fig = Figure()
    ax = fig.subplots()

    beliefs_a = [agent.a_expectations() for agent in model.agents]
    beliefs_b = [agent.b_expectations() for agent in model.agents]

    ax.hist(beliefs_a, bins=33, range=(0, 1),
            color="#2D6A4F", alpha=0.7, label="Belief in A")
    ax.hist(beliefs_b, bins=33, range=(0, 1),
            color="#C0392B", alpha=0.7, label="Belief in B")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Expected success rate")
    ax.set_ylabel("Number of agents")
    ax.set_title("Distribution of agent beliefs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    solara.FigureMatplotlib(fig)

def generate_results_zip(model):
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:

        # 1. Metriche del modello nel tempo (CSV)
        try:
            model_df = model.datacollector.get_model_vars_dataframe()
            zf.writestr("model_metrics.csv", model_df.to_csv())
        except ValueError:
            pass

        # 2. Metriche per agente nel tempo (CSV)
        try:
            agent_df = model.datacollector.get_agent_vars_dataframe()
            zf.writestr("agent_metrics.csv", agent_df.to_csv())
        except (ValueError, KeyError):
            pass

        # 3. Parametri della run (JSON)
        params = {
            "n_agents": model.num_agents,
            "graph": model.graph,
            "a_objective": model.a_objective,
            "b_objective": model.b_objective,
            "max priors": model.max_priors,
            "step_pulls": model.step_pulls,
            "dynamic": model.dynamic,
            "criticism": model.criticism,
            "inertia": list(model.agents)[0].inertia,
            "theory_threshold": list(model.agents)[0].theory_threshold,
            "seed": model.seed,
            "total_rounds": model.round_counter,
        }
        zf.writestr("parameters.json", json.dumps(params, indent=2))

        # 4. Sommario leggibile (TXT)
        conv_labels = {
            0: "No consensus reached",
            1: "Consensus on correct hypothesis (A)",
            2: "Consensus on wrong hypothesis (B)"
        }
        eva = (model.experiments_results_a["successes"] /
               model.experiments_results_a["trials"]
               if model.experiments_results_a["trials"] > 0 else 0)
        evb = (model.experiments_results_b["successes"] /
               model.experiments_results_b["trials"]
               if model.experiments_results_b["trials"] > 0 else 0)

        summary = f"""Epistemic Bandit Model — Run Summary
            ======================================
            Total rounds:       {model.round_counter}
            Convergence:        {conv_labels[model.convergence_status]}
            Consensus round:    {model.consensus_round if model.consensus_round else "N/A"}
            Agents on A:        {model.count_state_a() * 100:.1f}%
            Agents on B:        {model.count_state_b() * 100:.1f}%
            Empirical p(A):     {eva:.4f}
            Empirical p(B):     {evb:.4f}
            """
        zf.writestr("summary.txt", summary)

    buffer.seek(0)
    return buffer.read()



epistemic_ABM = Bandit()


SpaceViz = make_space_component(
    agent_portrayal, backend="matplotlib", layout_alg = nx.shell_layout, layout_kwargs={"scale": 1.0}
)




@solara.component
def Page():
    solara.Title("Epistemic Bandit Model")
    solara.lab.theme.themes.light.primary = "#2D6A4F"
    solara.lab.theme.themes.dark.primary  = "#2D6A4F" 
    solara.Style("""
        /* Nascondiamo il testo numerico originale (0, 1, 2) ma teniamo il pulsante */
        .v-tab {
            font-size: 0 !important;
        }
        
        /* Definiamo il nuovo nome per il Tab 0 */
        .v-tab:nth-child(2)::after {
            content: "Model";
            font-size: 14px !important; /* Ripristiniamo la dimensione del font */
            text-transform: uppercase;
            font-weight: 500;
        }

        /* Definiamo il nuovo nome per il Tab 1 */
        .v-tab:nth-child(3)::after {
            content: "Agents";
            font-size: 14px !important;
            text-transform: uppercase;
            font-weight: 500;
        }
    """)    

    viz = SolaraViz(
    epistemic_ABM,
    model_params=model_params,
    components=[SpaceViz, (deviation_plot, 0), (agent_stats, 0), (experiment_stats, 0), (belief_histogram, 1), (agents_table, 1), (aggregate_metrics, 1)],
    name="Epistemic Bandit ABM")

    
