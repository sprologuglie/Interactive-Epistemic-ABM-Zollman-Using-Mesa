import pytest
from scripts.model import Bandit

# Functions tests

def test_step():
    model = Bandit(seed=42)
    model.step()
    assert model.round_counter == 1

def test_convergence_a():
    model = Bandit(a_objective=0.9, b_objective=0.1, seed=42)
    for _ in range(500):
        model.step()
    assert model.check_convergence() == 1

def test_convergence_b():
    model = Bandit(a_objective=0.1, b_objective=0.9, seed=42)
    for _ in range(500):
        model.step()
    assert model.check_convergence() == 2

def test_belief():
    model = Bandit(n=10, seed=42)
    model.step()
    assert model.count_state_a() + model.count_state_b() == pytest.approx(1.0)

def test_experiments_round():
    model = Bandit(n=10, seed=42)
    model.step()
    assert model.experiments_round_results_a["trials"] + model.experiments_round_results_b["trials"] > 0

def test_experiments_total():
    model = Bandit(n=10, seed=42)
    model.step()
    assert model.experiments_results_a["trials"] + model.experiments_results_b["trials"] > 0

def test_experiments_sum():
    model = Bandit(n=10, seed=42)
    model.step()
    a_1 = model.experiments_round_results_a["trials"]
    b_1 = model.experiments_round_results_b["trials"]
    model.step()
    a_2 = model.experiments_round_results_a["trials"]
    b_2 = model.experiments_round_results_b["trials"]
    assert a_1 + a_2 == model.experiments_results_a["trials"]
    assert b_1 + b_2 == model.experiments_results_b["trials"]

def test_datacollector():
    model = Bandit(seed=42)
    for _ in range(20):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert len(df) == 20
    assert "Avg. A expectation" in df.columns
    assert "Avg. B expectation" in df.columns
    assert "Convergence Round" in df.columns
    assert "Correct Convergence" in df.columns

def test_consensus_round():
    model = Bandit(n=10, a_objective=0.9, b_objective=0.3, seed=42)
    for _ in range(100):
        model.step()
    assert model.consensus_round is not None
    assert model.consensus_round >= 1
    assert model.consensus_round <= model.round_counter

# Model variables tests

def test_number_agents():
    model = Bandit(n=5, seed=42)
    assert model.num_agents == 5

def test_graphs():
    with pytest.raises(ValueError):
        Bandit(n=5, graph="error", seed=42)
    
def test_cycle():
    cycle = Bandit(n=7, graph="cycle", seed=42)
    assert len(list(cycle.grid.all_cells.cells)) == 7

def test_complete():
    complete = Bandit(n=7, graph="complete", seed=42)
    assert len(list(complete.grid.all_cells.cells)) == 7

def test_wheel():
    wheel = Bandit(n=7, graph="wheel", seed=42)
    assert len(list(wheel.grid.all_cells.cells)) == 7

def test_seed():
    model_1 = Bandit(seed=95)
    model_2 = Bandit(seed=95)

    for _ in range(10):
        model_1.step()
        model_2.step()
    
    assert model_1.convergence_status == model_2.convergence_status
    assert model_1.count_state_a() == model_2.count_state_a()

def test_step_pulls():
    model = Bandit(n=10, step_pulls=100, seed=42)
    model.step()
    assert model.experiments_results_a["trials"] + model.experiments_results_b["trials"] == 100 * 10

def test_max_priors():
    model = Bandit(n=5, max_priors=10, seed=42)
    for a in model.agents:
        assert a.priors["a_alpha"] > 0 and a.priors["a_alpha"] <= 10
        assert a.priors["b_alpha"] > 0 and a.priors["b_alpha"] <= 10
        assert a.priors["a_beta"] > 0 and a.priors["a_beta"] <= 10
        assert a.priors["b_beta"] > 0 and a.priors["b_beta"] <= 10


def test_dynamic():
    model = Bandit(n=8, a_objective=0.5, b_objective=0.5, dynamic=1, seed=42)
    for _ in range(4):
        model.step()
    for a in model.agents:
        assert a.a_objective > 0.5
        assert a.b_objective < 0.5

def test_inertia():
    model = Bandit(n=10, a_objective=0.9, b_objective=0.3, inertia=0, seed=42)
    inertia = Bandit(n=10, a_objective=0.9, b_objective=0.3, inertia=10, seed=42)
    for _ in range(100):
        model.step()
        inertia.step()
    assert model.consensus_round is not None
    assert inertia.consensus_round is not None
    assert inertia.consensus_round > model.consensus_round    

def test_critical_interaction():
    model = Bandit(n=8, a_objective=0.5, b_objective=0.5, criticism=True, seed=42)
    for _ in range(500):
        model.step()
    a_objective_sum = sum(a.a_objective for a in model.agents)
    b_objective_sum = sum(a.b_objective for a in model.agents)
    assert a_objective_sum > 0.5 * 8
    assert b_objective_sum < 0.5 * 8

def test_theory_threshold():
    model = Bandit(n=10, a_objective=0.6, b_objective=0.4, theory_threshold=0, seed=42)
    tt = Bandit(n=10, a_objective=0.6, b_objective=0.4, theory_threshold=0.2, seed=42)
    for _ in range(100):
        model.step()
        tt.step()
    assert model.consensus_round is not None
    assert tt.consensus_round is not None
    assert tt.consensus_round > model.consensus_round
    
# Edge cases 
   
def test_small_n():
    complete = Bandit(n=2, graph="complete", seed=42)
    for _ in range(10):
        complete.step()
    assert complete.round_counter == 10

    cycle = Bandit(n=3, graph="cycle", seed=42)
    for _ in range(10):
        cycle.step()
    assert cycle.round_counter == 10

    wheel = Bandit(n=4, graph="wheel", seed=42)
    for _ in range(10):
        wheel.step()
    assert wheel.round_counter == 10