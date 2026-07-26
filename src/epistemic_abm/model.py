from __future__ import annotations

from typing import Literal

import mesa
import networkx as nx
import numpy as np
from mesa.discrete_space import Network

from .agent import Scientist

###                 MODEL                   ####
     
def _count_belief_a(model):
    """Average expected success rate of theory A across all agents."""
    return np.mean([agent.a_expectations for agent in model.agents])

       
def _count_belief_b(model):
    """Average expected success rate of theory B across all agents."""
    return np.mean([agent.b_expectations for agent in model.agents])


def _get_a_objective_probability(model):
    return np.mean([a.a_objective for a in model.agents]) 

def _get_b_objective_probability(model):
    return np.mean([a.b_objective for a in model.agents])

def _convergence_round(model):
    return model.consensus_round 

def _correct_convergence(model):
    return sum(1 for a in model.agents if a.state == "a") == model.num_agents


class Bandit(mesa.Model):
    """Epistemic Bandit model of scientific inquiry in a network of agents.

    Implements the Zollman (2010) bandit framework: agents choose between
    two theories (A and B) by running experiments and sharing evidence with
    their network neighbours.
    """

    def __init__(
        self,
        n: int = 10,
        a_objective: float = 0.5,
        b_objective: float = 0.499,
        max_priors: float = 4,
        graph: Literal["complete", "wheel", "cycle"] = "complete",
        theory_threshold: float = 0,
        step_pulls: int = 1000,
        dynamic: int | None = None,
        criticism: bool | None = None,
        inertia: int = 0,
        batch_mode: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialise the Epistemic Bandit model.

        Parameters
        ----------
        n : int
            Number of scientist agents (default 10).
        a_objective : float
            True success rate of theory A; must be >= b_objective (default 0.5).
        b_objective : float
            True success rate of theory B (default 0.499).
        max_priors : float
            Upper bound for initial Beta prior parameters, drawn uniformly
            from (0, max_priors) for each agent (default 4).
        graph : {"complete", "wheel", "cycle"}
            Network topology connecting agents (default "complete").
        theory_threshold : float
            Minimum belief gap required before an agent switches theories
            (default 0).
        step_pulls : int
            Number of binomial draws per experiment per step (default 1000).
        dynamic : int | None
            If set, objective probabilities shift toward their limits every
            ``dynamic`` rounds. ``None`` disables this feature (default None).
        criticism : bool | None
            If ``True``, enables critical interaction between neighbouring
            agents (default None).
        inertia : int
            Rounds an agent must prefer the alternative theory before
            switching (default 0).
        seed : int | None
            Random seed for reproducibility (default None).
        """

        super().__init__(rng=seed)
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
        self.theory_threshold = theory_threshold
        
        # Defining the graph type
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
        if batch_mode:
            self.datacollector = mesa.DataCollector(model_reporters={"Convergence Round": _convergence_round, "Correct Convergence": _correct_convergence})
        else:
            self.datacollector = mesa.DataCollector(
                model_reporters={"Avg. A expectation": _count_belief_a, "A objective probability": _get_a_objective_probability, "Avg. B expectation": _count_belief_b, "B objective probability": _get_b_objective_probability, "Convergence Round": _convergence_round, "Correct Convergence": _correct_convergence},
                agent_reporters={"Belief_A": "a_expectations", "Belief_B": "b_expectations", "State": "state"}
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
            if a.experiment_result is None:
                continue
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
        elif sum(1 for a in self.agents if a.state == "b") == self.num_agents:
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

        if (((not self.dynamic) and self.convergence_status != 0) or (self.dynamic and self.convergence_status == 1)) and self.round_counter > (self.consensus_round + 500):
            self.running = False
