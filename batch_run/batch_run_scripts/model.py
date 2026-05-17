import mesa
from mesa.discrete_space import Network
import networkx as nx

from .agent import Scientist

###                 MODEL                   ####
def Convergence_Round(model):
    if model.consensus_round == None:
        return "No consensus"
    else: return model.consensus_round

def Correct_Convergence(model):
    if sum(1 for a in model.agents if a.state == "a") == model.num_agents:
        return True
    else: return False

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
        self.theory_threshold = theory_threshold
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
        else:
            raise ValueError(f"Unknown network type '{graph}'. Use 'complete', 'wheel', or 'cycle'.")
        # Create agents
        Scientist.create_agents(
            model=self, n=n, cell=list(self.grid.all_cells.cells), a_objective = self.a_objective, b_objective = self.b_objective, max_priors = max_priors, theory_threshold = theory_threshold, inertia = inertia, step_pulls = step_pulls, dynamic = dynamic)
    
        # Instantiate DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Convergence Round": Convergence_Round,
                "Correct Convergence": Correct_Convergence    
            }
        )

        self.round_counter = 0
        self.consensus_round = None
        self.check_previous_conv = 0


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

        if self.dynamic != None:
            self.agents.do("Update_Objectives")
        
        if self.criticism == True:
            self.agents.do("critical_interaction")
        
        self.agents.do("update")
        self.agents.do("clean_results")

        self.round_counter += 1
        self.Check_Convergence()
        self.Get_Convergence_Round()
        
