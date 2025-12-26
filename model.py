import mesa
from mesa.discrete_space import Network, FixedAgent, CellCollection
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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

def Count_Belief(model):
    """Function for calculating the average distance from objective values of hypotheses between agents"""
    agents_a_dev = Count_Belief_a(model)/model.a_objective
    agents_b_dev = Count_Belief_b(model)/model.b_objective

    return (agents_a_dev + agents_b_dev) / 2



class Bandit(mesa.Model):
    """Model"""

    def __init__(
            self, 
            n=10,
            a_objective = .5, 
            b_objective = .4999, 
            max_priors = 4,
            graph = "complete",
            stubborness_range = 0,
                    ):

        super().__init__()
        self.num_agents = n
        self.a_objective = a_objective
        self.b_objective = b_objective
        self.stubborness_range = stubborness_range
        #Defining the graph type
        if graph == "complete":
            self.grid = Network(nx.complete_graph(n))
        elif graph == "wheel":
            self.grid = Network(nx.wheel_graph(n))
        elif graph == "cycle":
            self.grid = Network(nx.cycle_graph(n))
        # Create agents
        Scientist.create_agents(
            model=self, n=n, cell=list(self.grid.all_cells.cells), a_objective = a_objective, b_objective = b_objective, max_priors = max_priors, stubborness_range = stubborness_range)
    
        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Avg. deviance from objectve": Count_Belief, "Avg. A expectation": Count_Belief_a, "Avg. B expectation": Count_Belief_b}
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

       
    def Count_State_a(self):
        """Function for counting how may agents prefer to pull A"""
        return sum(1 for a in self.agents if a.state == "a")/sum(1 for _ in self.agents)

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


        return self.experiments_round_results_a, self.experiments_round_results_b
            
    def Update_Evidence(self):
        """Function for updating experiment results data"""
        self.experiments_results_a["successes"] += self.experiments_round_results_a["successes"]
        self.experiments_results_a["trials"] += self.experiments_round_results_a["trials"]
        self.experiments_results_b["successes"] += self.experiments_round_results_b["successes"]
        self.experiments_results_b["trials"] += self.experiments_round_results_b["trials"]          


    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)       
        self.agents.do("research")
        self.Count_Evidence()
        self.Update_Evidence()
        self.agents.do("update")