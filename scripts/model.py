import mesa
from mesa.discrete_space import Network
import numpy as np
import networkx as nx
from .agent import Scientist

###                 MODEL                   ####
     
def count_belief_a(model):
    """Funcion for counting the average expectation of a between agents"""
    agents_a_exp = [agent.priors["a_alpha"] / 
            (agent.priors ["a_alpha"] + agent.priors ["a_beta"]) for agent in model.agents]
    ma = np.mean(agents_a_exp)

    return ma

       
def count_belief_b(model):
    """Function for counting the average experctation of b between agents"""
    agents_b_exp = [agent.priors["b_alpha"] / 
            (agent.priors ["b_alpha"] + agent.priors ["b_beta"]) for agent in model.agents]
    mb = np.mean(agents_b_exp)

    return mb


def get_a_objective_probability(model):
    return np.mean([a.a_objective for a in model.agents]) 

def get_b_objective_probability(model):
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
            theory_threshold = 0,
            step_pulls = 1000,
            dynamic = None,
            criticism = None,
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
        self.graph = graph
        self.seed = seed
        self.max_priors = max_priors
        self.inertia = inertia
        #Defining the graph type
        if graph == "complete":
            self.grid = Network(nx.complete_graph(n), random=self.random)
        elif graph == "wheel":
            self.grid = Network(nx.wheel_graph(n), random=self.random)
        elif graph == "cycle":
            self.grid = Network(nx.cycle_graph(n), random=self.random)
        else:
            raise ValueError(f"Unknown network type '{graph}'. Use 'complete', 'wheel', or 'cycle'.")
        # Create agents
        Scientist.create_agents(
            model=self, n=n, cell=list(self.grid.all_cells.cells), a_objective = self.a_objective, b_objective = self.b_objective, max_priors = max_priors, theory_threshold = theory_threshold, inertia = inertia, step_pulls = step_pulls, dynamic = dynamic)
    
        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Avg. A expectation": count_belief_a, "A objective probability": get_a_objective_probability, "Avg. B expectation": count_belief_b, "B objective probability": get_b_objective_probability},
            agent_reporters={"Belief_A": lambda a: a.a_expectations(), "Belief_B": lambda a: a.b_expectations(), "State": "state"}
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
        self.convergence_status = 0

       
    def count_state_a(self):
        """Function for counting how may agents prefer to pull A"""
        return sum(1 for a in self.agents if a.state == "a")/self.num_agents
    
    def count_state_b(self):
        """Function for counting how may agents prefer to pull B"""
        return sum(1 for a in self.agents if a.state == "b")/self.num_agents

    def count_evidence(self):
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

            
    def update_evidence(self):
        """Function for updating experiment results data"""
        self.experiments_results_a["successes"] += self.experiments_round_results_a["successes"]
        self.experiments_results_a["trials"] += self.experiments_round_results_a["trials"]
        self.experiments_results_b["successes"] += self.experiments_round_results_b["successes"]
        self.experiments_results_b["trials"] += self.experiments_round_results_b["trials"]          
        
    
    def check_convergence(self):
        """Checks whether all agents pursue the same hypothesis"""
        
        if sum(1 for a in self.agents if a.state == "a") == self.num_agents:
            if self.consensus_round and self.check_previous_conv != 1:
                self.consensus_round = None
            self.check_previous_conv = 1
            return 1
        if sum(1 for a in self.agents if a.state == "b") == self.num_agents:
            if self.consensus_round and self.check_previous_conv != 2:
                self.consensus_round = None
            self.check_previous_conv = 2
            return 2
        else:
            self.consensus_round = None 
            self.check_previous_conv = 0
            return 0
    
    def get_convergence_round(self):
        """Get the round in which agents converged"""
        conv = self.check_convergence()
        self.convergence_status = conv
        if (conv == 1 or conv == 2) and not self.consensus_round:
            self.consensus_round = self.round_counter
        

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
               
        self.agents.do("research")

        self.count_evidence()
        self.update_evidence()

        if self.dynamic:
            self.agents.do("update_objectives")
        
        if self.criticism:
            self.agents.do("critical_interaction")
        
        self.agents.do("update")
        self.agents.do("clean_results")

        self.round_counter += 1
        self.get_convergence_round()