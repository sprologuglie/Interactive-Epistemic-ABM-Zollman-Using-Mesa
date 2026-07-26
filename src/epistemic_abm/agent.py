from __future__ import annotations

import mesa
from mesa.discrete_space import FixedAgent

###                 AGENTS                  ###

class Scientist(FixedAgent):

    def __init__(
        self,
        model: mesa.Model,
        cell,
        a_objective: float,
        b_objective: float,
        max_priors: float,
        theory_threshold: float,
        inertia: int,
        step_pulls: int,
        dynamic: int | None,
    ) -> None:
        """Initialise a Scientist agent.

        Parameters
        ----------
        model : mesa.Model
            The parent Bandit model instance.
        cell :
            Network cell this agent occupies (assigned by Mesa).
        a_objective : float
            True success rate of theory A.
        b_objective : float
            True success rate of theory B.
        max_priors : float
            Upper bound for initial Beta prior parameters.
        theory_threshold : float
            Minimum belief gap required before switching theories.
        inertia : int
            Rounds the agent must prefer the alternative before switching.
        step_pulls : int
            Number of binomial draws per experiment per step.
        dynamic : int | None
            Interval (in rounds) for shifting objective probabilities;
            ``None`` disables dynamic objectives.
        """
        super().__init__(model)
        self.cell = cell
        self.a_objective = a_objective
        self.b_objective = b_objective
        self.max_priors = max_priors
        self.theory_threshold = theory_threshold
        self.inertia = inertia
        self.inertia_counter = 0
        self.step_pulls = step_pulls
        self.dynamic = dynamic

        #Prior beliefs of each agent
        epsilon = 1e-21
        self.priors = {
        "a_alpha": self.random.uniform(epsilon, self.max_priors),
        "a_beta": self.random.uniform(epsilon, self.max_priors),
        "b_alpha": self.random.uniform(epsilon, self.max_priors),
        "b_beta": self.random.uniform(epsilon, self.max_priors)
        }

        #Define whether the agents prefers to pull the A or B lever as it's state
        if self.a_expectations > self.b_expectations:
            self.state = "a"
        else:
            self.state = "b"
        

        self.dynamic_counter = 0
        
        self.experiment_result: tuple[int, int, int] | None = None
        

    # Computed properties: expected success rate for each hypothesis
    @property
    def a_expectations(self) -> float:
        """Expected success rate of theory A under current Beta priors."""
        return self.priors["a_alpha"] / (self.priors["a_alpha"] + self.priors["a_beta"])

    @property
    def b_expectations(self) -> float:
        """Expected success rate of theory B under current Beta priors."""
        return self.priors["b_alpha"] / (self.priors["b_alpha"] + self.priors["b_beta"])
    
    
    def research(self):
        """Research behaviour: performing experiments"""

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

    
    def update(self):
        """Update behaviour: updating expectations based on experimental results"""
        # Update beliefs based on OWN results
        if self.experiment_result is None:
            return
        pull, success, trial = self.experiment_result

        if pull == 1:
            self.priors["a_alpha"] += success
            self.priors["a_beta"] += trial - success
        else:
            self.priors["b_alpha"] += success
            self.priors["b_beta"] += trial - success

        #Update beliefs based on NEIGHBORS results
        for neighbor in self.cell.neighborhood.agents:
            if neighbor.experiment_result is None:
                continue
            pull, success, trial = neighbor.experiment_result

            if pull == 1:
                self.priors["a_alpha"] += success
                self.priors["a_beta"] += trial - success
            else:
                self.priors["b_alpha"] += success
                self.priors["b_beta"] += trial - success
            
        #Updating preferences for experimentations (include theory_threshold and inertia)    
        if self.state == "a":
            if (self.a_expectations + self.theory_threshold) > self.b_expectations:
                self.state = "a"
                self.inertia_counter = 0
            else:
                self.inertia_counter += 1
                if self.inertia_counter >= self.inertia:
                    self.state = "b"
        
        else:
            if (self.b_expectations + self.theory_threshold) > self.a_expectations:
                self.state = "b"
                self.inertia_counter = 0
            else:
                self.inertia_counter += 1
                if self.inertia_counter >= self.inertia:
                    self.state = "a"
    
    def update_objectives(self):
        """Shift objective probabilities toward their limits every `dynamic` rounds.

        a_objective moves toward 1, b_objective toward 0,
        each by 1/1000 of the remaining distance.
        """
        if self.dynamic_counter < self.dynamic:
            self.dynamic_counter += 1
        else:
            self.dynamic_counter = 0
            self.a_objective += (1- self.a_objective) / 1000
            self.b_objective += (0- self.b_objective) / 1000
        
    def critical_interaction(self):
        """Slightly modify the objective values if neighbors provide more covincing evidence for the competing hypothesis"""    
        if self.experiment_result is None:
            return
        pull, _, _ = self.experiment_result

        for neighbor in self.cell.neighborhood.agents:
            if neighbor.experiment_result is None:
                continue
            neigh_pull, neigh_success, neigh_trial = neighbor.experiment_result

            if pull == 1 and neigh_pull != pull and neigh_success / neigh_trial > self.b_expectations:
                self.a_objective += (1- self.a_objective) / 1000

            elif pull == 2 and neigh_pull != pull and neigh_success / neigh_trial > self.a_expectations:
                    self.b_objective += (0 - self.b_objective) / 1000
    
    def clean_results(self):
        self.experiment_result: tuple[int, int, int] | None = None