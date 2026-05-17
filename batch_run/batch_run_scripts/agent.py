from mesa.discrete_space import FixedAgent

###                 AGENTS                  ###

class Scientist(FixedAgent):

    def __init__(self, model, cell, a_objective, b_objective, max_priors, theory_threshold, inertia, step_pulls, dynamic):
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
            
        #Updating preferences for experimentations (include theory_threshold and inertia)    
        if self.state == "a":
            if (self.a_expectations() + self.theory_threshold) > self.b_expectations():
                self.state = "a"
                self.inertia_counter = 0
            else:
                self.inertia_counter += 1
                if self.inertia_counter >= self.inertia:
                    self.state = "b"
        
        else:
            if (self.b_expectations() + self.theory_threshold) > self.a_expectations():
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