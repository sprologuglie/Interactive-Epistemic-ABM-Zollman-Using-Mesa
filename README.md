# Epistemic ABM (simple interactive replica)
**Work in progress**

A **minimal** repository that provides:

- a small **agent-based model** (epistemic network + two-armed bandit)
- an **interactive** dashboard via **Mesa + Solara**
- a **batch run** script and notebook to run many simulations headlessly
- a **replication of the studies** conducted on this model: **Zollman (2010), Rosenstock et al. (2017), Frey and Šešelja (2020)**

This repo is intentionally lightweight: just scripts you can run.

## Dashboard Quickstart

### How it works
You can modify the parameters trough the panel on the left, then press restart to apply changes. Then let the model run continuously with the play button or for a single step with the step button.

**NB If you don't press Restart before starting the model the parameters won't change**.

For more info about the model and the parameters read the next sections.


### Try it in your browser (no install)

- **Interactive notebook (Binder):** 
Click on Launch Binder, then wait for the server to be started. **Run the model cell and click on the fullscreen button of the output.** Follow the instruction above (### How it works)

  open `interactive_model.ipynb` 

  [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sprologuglie/Interactive-Epistemic-ABM-Zollman-Using-Mesa/main?urlpath=lab/tree/interactive_model.ipynb)


### Run locally (recommended for the interactive dashboard)

Download the repo and on you terminal write

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
solara run interactive_model.py
```
Solara will print a local URL (typically `http://localhost:8765`). Open it in your browser. Follow the instruction above (How it works)


## Batch runs (headless)

### Try it in your browser (no install)

- **Batch runs notebook (Binder):** 
Click on Launch Binder, then wait for the server to be started.
    
  [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sprologuglie/Interactive-Epistemic-ABM-Zollman-Using-Mesa/main?urlpath=lab/tree/batch_run/batch_run.ipynb)

  Once Binder has started the server, follow the instruction below (Notebook)

### Script
Modify the parameters, the number of iterations and the max_steps, the kind of graph desired and then batch run the simulations. Outputs will be shown in a graph and stored in a csv file.
Default settings reproduce the Zollman effect.

### Notebook
If you prefer, you may use a Jupiter notebook to batch run simulations. First run the model cell. After that, modify the parameters, the number of iterations and the max_steps, the kind of graph desired and then batch run the simulations. Outputs will be shown in a graph, to store them in a csv file run the last cell. Feel free to duplicate the batch trun cell, change parameters and repeat the process using different combinations of parameters.
Default settings reproduce the Zollman effect.

## What kind of model this is (and how it connects to the three papers)
This is a networked two-armed bandit model of scientific inquiry:

  - There are two rival “theories/methods”: A (true) and B (false).

  - Each agent (a “Scientist”) has Bayesian beliefs about each theory’s success rate, represented with Beta priors (α, β). Updating is done by adding successes to α and failures to β (conjugacy), and the expected success rate is α/(α+β).

  - Agents are myopic exploiters: in each round they choose whichever action currently has the higher expected payoff and performs some experiments (Bernoulli trials) on the preffered arm of the bandit.

  - Agents sit on a communication network (which can be a complete, wheel or cycle graph), and after each round they update their beliefs using their own evidence + their neighbors’ evidence.

### General structure and how one simulation run works
Entities:

  - Model: Bandit (Mesa Model)

  - Agents: Scientist (Mesa FixedAgent) placed on a Network grid (NetworkX graph) 


Agent belief representation (per theory)

  - Each Scientist holds four Beta parameters:

    - for A: a_alpha, a_beta
    - for B: b_alpha, b_beta

Expected success estimates:

  - E[A] = a_alpha / (a_alpha + a_beta)

  - E[B] = b_alpha / (b_alpha + b_beta)

Choice rule (state)

  - Each agent has a discrete state in { "a", "b" }: Choose A if E[A] > E[B], else choose B. 

One model step (one “round”):

  - Collect data (datacollector)

  -  All agents research:

      - pick the currently preferred arm (A or B),

      - generate results: success ~ Binomial(n=step_pulls, p=objective_prob) (objective_prob depends on which arm was pulled).

      - If enabled: dynamic success update (Update_Objectives)

      - If enabled: critical interaction update (critical_interaction)

  -  All agents update beliefs:

      - incorporate own results,

      - incorporate neighbors’ results,

      - possibly switch state, subject to theory-threshold and inertia.

  - Agents clean their stored result tuple

  - Model updates round counter and checks convergence (everyone on A or everyone on B), storing the consensus round

### Model-level parameters

These are the knobs you pass when constructing the model. 

- n (default 10)
  - Meaning: number of scientists (agents).

- a_objective (default 0.5)
  - Meaning: the “ground truth” success probability when pulling arm A (unless later modified by dynamic / criticism).
  - Role: sets how good theory A actually is, i.e., the Bernoulli/binomial parameter for A-generated data. 

- b_objective (default 0.499)
  - Meaning: ground truth success probability for arm B (again, unless modified).
  - Role: together with a_objective, determines how easy it is to tell the theories apart. If very close, inquiry is “hard”.

- max_priors (default 4)
  - Meaning: upper bound for the initial random Beta parameters α and β (drawn uniformly from (ε, max_priors)).
  - Role: controls how “strong/extreme” initial priors can be on average. Initial α/β ranges affect resistance to early evidence and diversity retention.
  
- graph (default "complete"; allowed: "complete", "wheel", "cycle")
  - Meaning: network topology used for communication.
  - Role: this is the central “connectedness” variable in all three papers (complete vs sparser graphs).

- step_pulls (default 1000)
  - Meaning: number of Bernoulli trials per round per agent (the n in the Binomial(n, p)).
  - Role: controls evidence volume per round. Smaller values make inquiry harder and amplify the chance of misleading streaks. It is one of Rosenstock’s key drivers of when connectivity matters.

- theory_threshold (default None)
  - Meaning: margin in which the agents don't switch theory even if the competing have an higher expectation
  - Role: implements a switching margin / indifference interval: an agent sticks with the current theory unless the alternative is better by more than the threshold (in code: it compares E[current] + threshold > E[other]). This corresponds to Frey–Šešelja’s “interval within which theories count as equally good.”

- dynamic (default None)
  - Meaning: the number of rounds until agents do Update_Objectives. If it is None, the update does not take place.
  - Role: implements dynamic epistemic success: every specified number of rounds A is nudged upward toward 1 and B downward toward 0 (small increments of size ≈1/1000 of the remaining distance). This parallels Frey–Šešelja’s move from static to dynamic success assumptions.

- criticism (default False)
  - Meaning: toggles the agents' critical_interaction. If it is False, they don't.
  - Role: implements a form of critical interaction inspired by Frey–Šešelja: neighbors’ evidence in favour of the competing theory can trigger a small change to the objective parameters (same as for dynamic).

- inertia (default 0)
  - Meaning: number of consecutive “rounds of being outvoted by evidence” required before an agent actually switches.
  - Role: implements Frey–Šešelja’s “rational inertia”: agents don’t instantly flip when the other arm looks (slightly) better; they require sustained pressure (for a sufficient number of rounds).

### Other crucial variables to understand the model
Inside each Scientist

  - priors: dict of four Beta parameters (a_alpha, a_beta, b_alpha, b_beta). 

  - state: current preferred theory (“a” or “b”). 

  - experiment_result: a tuple (pull, success, trial) stored until updating is done; then reset. 

  - inertia_counter: how many consecutive rounds the agent has been “tempted” to switch but hasn’t yet. 

  - dynamic_counter: counts rounds up to the specified number of rounds to trigger objective drift. 

  - critical_interaction_counter: counts rounds up to the specified number of rounds to trigger critical interaction.


Inside the Bandit model

  - round_counter: number of elapsed rounds. 

  - consensus_round: first round when everyone chooses the same arm (stored; reset if consensus flips). 

  - check_previous_conv: tracks whether last consensus was on A or B to detect changes.

  - convergence_status: Store the current convergence status -> 0 if no consensus, 1 if consensus on A and 2 if consensus on B

### Statistics in the dashboard

In the graph of the left the nodes represent the agents and the edges the neighbors to which they are connected. The color of the node represents the the state of the agent (green if they prefer A, red if they prefer B). <br>

Below, the percentage of agents believing in A and B is present, togheter with and indication of the Consesus and the round in which the current consensus was established. <br>

In the graph on the right the *Avg A expectation* represent the mean of the expected outcome of A for all the agents. This metric shows how on average agents are far from the actual value of A, plotted as *A objective probability*. The same goes for B. <br>

Below the results of the experiments of each round is shown, as well as the total evidence for both theories and their expectations based on evidence only. <br>

## Replications

Replications of three studies conducted on this model are presented in `replications/`. The replication were performed using an older version of the model, were there was no control over the rounds for the dynamic update of objective probabilities (it was a switch to trigger it every 100 round, like in Frey & Šešelja (2020)) and no control over the exact theory threshold (which again was necessarily 0.1 as in Frey & Šešelja (2020)). 

## Repository layout

```
interactive_model.py                 # Interactive entrypoint (use with `solara run interactive_model.py`)
interactive_model.ipynb              # Used to produce the interactive model inside a Jupyter Notebook
batch_run/
  batch_run_scripts/                 # Scripts for batch run (include different metrics, comparable to the ones used in the papers)
    agent.py
    model.py
  batch_run.ipynb                    # Notebook for batch running
  batch_run.py                       # Script for bacth running
scripts/
  agent.py                           # Agents (Scientists)
  model.py                           # ABM (Bandit model)
  app.py                             # Solara visualization
replications/
  Zollman_replication.ipynb          # Replication of Zollman main result
  Rosenstocketal_replication.ipynb   # Replication of Rosenstock et al. robustness tests
  Frey_Seselja_replication.ipynb     # Replication of Frey and Šešelja variations on Zollman's model
README.md
REFERENCES.md
CONTRIBUTING.md
LICENSE
requirements.txt
runtime.txt
.gitignore
```

## License
MIT — see `LICENSE`.

## Citation
If you use this code in academic work, please cite the original papers listed in `REFERENCES.md` and also link to this repository.
