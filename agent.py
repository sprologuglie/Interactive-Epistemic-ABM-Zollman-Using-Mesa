import mesa
from mesa.discrete_space import Network, FixedAgent, CellCollection
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

###                 AGENTS                  ###
class Scientist(FixedAgent):

    def __init__(self, model, cell, a_objective, b_objective, max_priors, stubborness_range):
        super().__init__(model)
        self.cell = cell
        self.a_objective = a_objective
        self.b_objective = b_objective
        self.max_priors = max_priors
        self.stubborness_range = stubborness_range

        #Prior beliefs of each agent
        self.priors = {
        "a_alpha": random.randint(1, self.max_priors),
        "a_beta": random.randint(1, self.max_priors),
        "b_alpha": random.randint(1, self.max_priors),
        "b_beta": random.randint(1, self.max_priors),
        }

        #Define whether the agents prefers to pull the A or B lever as it's state
        if self.a_expectations() > self.b_expectations():
            self.state = "a"
        else:
            self.state = "b"
        
        #Define the level of stubborness of the agent
        self.stubborness = .1 * random.randint(0, self.stubborness_range)
        
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


        #Choose action based of belief
        if self.state == "a":
            pull = 1
            current_objective_prob = self.a_objective
        else:
            pull = 2
            current_objective_prob = self.b_objective
        
        #Performing experiment (sampling from a binomial distribution)
        if random.random() < current_objective_prob:
            success = 1
        else: 
            success = 0

        self.experiment_result = (pull, success, 1)
        
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
            
        #Updating preferences for experimentations (include stubborness)    
        if self.state == "a":
            if (self.a_expectations() + self.stubborness) > self.b_expectations():
                self.state = "a"
            else:
                self.state = "b"
        
        else:
            if (self.b_expectations() + self.stubborness) > self.a_expectations():
                self.state = "b"
            else:
                self.state = "a"