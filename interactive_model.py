# TO RUN THE INTERACTIVE DASHBOARD USE: solara run interactive_model.py
# For more info see the README

import mesa
from mesa.discrete_space import Network, FixedAgent
import numpy as np
import networkx as nx



###                 AGENTS                  ###

class Scientist(FixedAgent):

    def __init__(self, model, cell, a_objective, b_objective, max_priors, theory_treshold, inertia, step_pulls, dynamic):
        super().__init__(model)
        self.cell = cell
        self.a_objective = a_objective
        self.b_objective = b_objective
        self.max_priors = max_priors
        self.theory_treshold = theory_treshold
        self.inertia = inertia
        self.inertia_counter = 0
        self.step_pulls = step_pulls
        self.dynamic = dynamic

        #Prior beliefs of each agent
        epsilon = .000000000000000000001
        self.priors = {
        "a_alpha": self.random.uniform(epsilon, self.max_priors),
        "a_beta": self.random.uniform(epsilon, self.max_priors),
        "b_alpha": self.random.uniform(epsilon, self.max_priors),
        "b_beta": self.random.uniform(epsilon, self.max_priors)
        }

        #Define whether the agents prefers to pull the A or B lever as it's state
        if self.a_expectations() > self.b_expectations():
            self.state = "a"
        else:
            self.state = "b"
        

        self.dynamic_counter = 0
        
        self.experiment_result = (0, 0, 0)
        

    #Funcions for calculating expectations for each hypotheses
    def a_expectations(self):
        a_exp = self.priors["a_alpha"] / (self.priors ["a_alpha"] + self.priors ["a_beta"])
        return a_exp

    def b_expectations(self):
        b_exp = self.priors["b_alpha"] / (self.priors ["b_alpha"] + self.priors ["b_beta"])
        return b_exp
    
    #Research behaviour: performing experiments
    def research(self):

        pull, success, trial = self.experiment_result

        #Choose action based of belief
        if self.state == "a":
            pull = 1
            current_objective_prob = self.a_objective
        else:
            pull = 2
            current_objective_prob = self.b_objective
        
        #Performing experiment (sampling from a binomial distribution)
        success = self.rng.binomial(n=self.step_pulls, p=current_objective_prob)
        
        trial = self.step_pulls

        self.experiment_result = (pull, success, trial)
        
        return self.experiment_result

    #Update behaviour: updating expectations based non experimental results
    def update(self):
        # Update beliefs based on OWN results
        pull, success, trial = self.experiment_result

        if pull == 1:
            self.priors["a_alpha"] += success
            self.priors["a_beta"] += trial - success
        else:
            self.priors["b_alpha"] += success
            self.priors["b_beta"] += trial - success

        #Update beliefs based on NEIGHBORS results
        for neighbor in self.cell.neighborhood.agents: 
            
            pull, success, trial = neighbor.experiment_result

            if pull == 1:
                self.priors["a_alpha"] += success
                self.priors["a_beta"] += trial - success
            else:
                self.priors["b_alpha"] += success
                self.priors["b_beta"] += trial - success
            
        #Updating preferences for experimentations (include theory_treshold and inertia)    
        if self.state == "a":
            if (self.a_expectations() + self.theory_treshold) > self.b_expectations():
                self.state = "a"
                self.inertia_counter = 0
            else:
                self.inertia_counter += 1
                if self.inertia_counter >= self.inertia:
                    self.state = "b"
        
        else:
            if (self.b_expectations() + self.theory_treshold) > self.a_expectations():
                self.state = "b"
                self.inertia_counter = 0
            else:
                self.inertia_counter += 1
                if self.inertia_counter >= self.inertia:
                    self.state = "a"
    
    def Update_Objectives(self):
        """Slightley modify the objective values to increase the one of the correct theory and diminish the one of the incorrect every 100 rounds"""
        if self.dynamic_counter < self.dynamic:
            self.dynamic_counter += 1
        else:
            self.dynamic_counter = 0
            self.a_objective += (1- self.a_objective) / 1000
            self.b_objective += (0- self.b_objective) / 1000
        
    def critical_interaction(self):
        """Slightly modify the objective values if neighbors provide more covincing evidence for the competing hypothesis"""
        pull, success, trial = self.experiment_result

        for neighbor in self.cell.neighborhood.agents: 
            neigh_pull, neigh_success, neigh_trial = neighbor.experiment_result

            if pull == 1:
                if neigh_pull != pull and neigh_success / neigh_trial > self.b_expectations():
                    self.a_objective += (1- self.a_objective) / 1000

            if pull == 2:
                if neigh_pull != pull and neigh_success / neigh_trial > self.a_expectations():
                    self.b_objective += (0 -self.b_objective) / 1000
    
    def clean_results(self):
        self.experiment_result = (0, 0, 0)
            
        

    
###                 MODEL                   ####
     
def Count_Belief_a(model):
    """Funcion for counting the average expectation of a between agents"""
    agents_a_exp = [agent.priors["a_alpha"] / 
            (agent.priors ["a_alpha"] + agent.priors ["a_beta"]) for agent in model.agents]
    ma = np.mean(agents_a_exp)

    return ma

       
def Count_Belief_b(model):
    """Function for counting the average experctation of b between agents"""
    agents_b_exp = [agent.priors["b_alpha"] / 
            (agent.priors ["b_alpha"] + agent.priors ["b_beta"]) for agent in model.agents]
    mb = np.mean(agents_b_exp)

    return mb


def Get_A_Objective(model):
    return np.mean(list(a.a_objective for a in model.agents))

def Get_B_Objective(model):
    return np.mean(list(a.b_objective for a in model.agents))



class Bandit(mesa.Model):
    """Model"""

    def __init__(
            self, 
            n=10,
            a_objective = .5, 
            b_objective = .499, 
            max_priors = 4,
            graph = "complete",
            theory_treshold = False,
            step_pulls = 1000,
            dynamic = False,
            criticism = False,
            inertia = 0,
            seed = None
                    ):

        super().__init__(seed=seed)
        self.num_agents = n
        self.a_objective = a_objective
        self.b_objective = b_objective
        self.step_pulls = step_pulls
        self.dynamic = dynamic
        self.criticism = criticism
        #Defining the graph type
        if graph == "complete":
            self.grid = Network(nx.complete_graph(n), random=self.random)
        elif graph == "wheel":
            self.grid = Network(nx.wheel_graph(n), random=self.random)
        elif graph == "cycle":
            self.grid = Network(nx.cycle_graph(n), random=self.random)
        else : print("Uknown network type: please use ['complete', 'wheel', 'cycle']")
        # Create agents
        Scientist.create_agents(
            model=self, n=n, cell=list(self.grid.all_cells.cells), a_objective = self.a_objective, b_objective = self.b_objective, max_priors = max_priors, theory_treshold = theory_treshold, inertia = inertia, step_pulls = step_pulls, dynamic = dynamic)
    
        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Avg. A expectation": Count_Belief_a, "A objective probability": Get_A_Objective, "Avg. B expectation": Count_Belief_b, "B objective probability": Get_B_Objective}
        )
        #Create dictionaries for total experiments results
        self.experiments_results_a = {
            "successes": 0,
            "trials": 0
        }
        self.experiments_results_b = {
            "successes": 0,
            "trials": 0
        }

        self.experiments_round_results_a = {
            "successes": 0,
            "trials": 0
        }
        self.experiments_round_results_b = {
            "successes": 0,
            "trials": 0
        }

        self.round_counter = 0
        self.consensus_round = None
        self.check_previous_conv = 0

       
    def Count_State_a(self):
        """Function for counting how may agents prefer to pull A"""
        return sum(1 for a in self.agents if a.state == "a")/sum(1 for _ in self.agents)
    
    def Count_State_b(self):
        """Function for counting how may agents prefer to pull B"""
        return sum(1 for a in self.agents if a.state == "b")/sum(1 for _ in self.agents)

    def Count_Evidence(self):
        """Function for collecting the experiments results"""

        self.experiments_round_results_a = {
            "successes": 0,
            "trials": 0
        }
        self.experiments_round_results_b = {
            "successes": 0,
            "trials": 0
        }

        for a in self.agents:
            action, success, trial = a.experiment_result
            if action == 1:
                self.experiments_round_results_a["successes"] += success
                self.experiments_round_results_a["trials"] += trial
            else:
                self.experiments_round_results_b["successes"] += success
                self.experiments_round_results_b["trials"] += trial

            
    def Update_Evidence(self):
        """Function for updating experiment results data"""
        self.experiments_results_a["successes"] += self.experiments_round_results_a["successes"]
        self.experiments_results_a["trials"] += self.experiments_round_results_a["trials"]
        self.experiments_results_b["successes"] += self.experiments_round_results_b["successes"]
        self.experiments_results_b["trials"] += self.experiments_round_results_b["trials"]          
        
    
    def Check_Convergence(self):
        """Checks whether all agents pursue the same hypothesis"""
        
        if sum(1 for a in self.agents if a.state == "a") == self.num_agents:
            if self.consensus_round != None and self.check_previous_conv != 1:
                self.consensus_round = None
            self.check_previous_conv = 1
            return 1
        if sum(1 for a in self.agents if a.state == "b") == self.num_agents:
            if self.consensus_round != None and self.check_previous_conv != 2:
                self.consensus_round = None
            self.check_previous_conv = 2
            return 2
        else:
            self.consensus_round = None 
            self.check_previous_conv = 0
            return 0
    
    def Get_Convergence_Round(self):
        """Get the round in which agents converged"""
        if (self.Check_Convergence() == 1 or self.Check_Convergence() == 2) and self.consensus_round == None:
            self.consensus_round = self.round_counter
        

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
               
        self.agents.do("research")

        self.Count_Evidence()
        self.Update_Evidence()

        if self.dynamic != None:
            self.agents.do("Update_Objectives")
        
        if self.criticism == True:
            self.agents.do("critical_interaction")
        
        self.agents.do("update")
        self.agents.do("clean_results")

        self.round_counter += 1
        self.Check_Convergence()
        self.Get_Convergence_Round()
        



###                 APP                 ###

import solara
import solara.lab
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_space_component
)
from mesa.visualization.components import AgentPortrayalStyle
from matplotlib.figure import Figure

from mesa.visualization.utils import update_counter    

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
        "type": "Checkbox",
        "label": "Critical interaction",
        "value": False
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
        f"STEP:<br>Number of experiments for A: {res_round_a["trials"]} with {res_round_a["successes"]} successes <br>Number of experiments for B: {res_round_b["trials"]} with {res_round_b["successes"]} successes<br>TOTAL:<br>Total number of experiments for A: {res_tot_a["trials"]} with {res_tot_a["successes"]} successes<br>Total number of experiments for B: {res_tot_b["trials"]} with {res_tot_b["successes"]} successes<br>Evidence based A = {eva}<br>Evidence based B = {evb}"
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

