# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Versions correspond to GitHub releases and Zenodo snapshots.

---

## [0.2.0]

### Added

**Model**
- `agent_reporters` in DataCollector (`Belief_A`, `Belief_B`, `State`) for per-agent time-series data
- `convergence_status` model attribute storing the result of `Check_Convergence()` at each step, avoiding repeated calls with side effects
- `batch_run` parameter added.

**Dashboard**
- Second tab "Agent Analytics" with per-agent belief table, aggregate diversity metrics (variance of A and B beliefs, Shannon entropy of A/B state distribution), and belief distribution histogram
- Download button exporting a `.zip` archive with model metrics CSV, agent metrics CSV, run parameters JSON, and plain-text summary
- `seed` parameter exposed in dashboard controls

**Batch run**
- Timestamped output filenames in `save_df_csv` to prevent overwriting previous results
- `ValueError` raised for unknown graph types instead of silent `print`

**Replications**
- `Zollman_replication.ipynb` now include a replication the second finding of the 2010 paper

**Documentation**
- `README.md` rewritten to reflect model changes and be more concise and clear.

### Fixed

- `Convergence_Round` reporter in batch run model returned mixed types (string `"No consensus"` or int); now returns `None` when no consensus is reached
- `test_seed`: was comparing return value of `Get_Convergence_Round()` (always `None`) instead of `model.consensus_round`
- `test_step_pulls` in batch run tests: was reading `agent.experiment_result` after `clean_results()` reset it to `(0,0,0)`; now checks `agent.step_pulls` attribute directly
- `epsilon` in agent prior initialisation: replaced unreadable literal with `1e-21`
- `criticism=None` default replaced with `False` for consistency in batch run parameters
- Binder functionality restored: `requirements.txt` fixed to include only necessary library


### Changed

#### agent.py
- a_expectations() and b_expectations() converted to @property with docstrings and return type annotations (-> float); updated all call sites
- critical_interaction(): second if pull == 2 converted to elif, eliminating an unnecessary check on mutually exclusive values
- Fixed update_objectives() docstring: removed incorrect reference to "every 100 rounds", now correctly describes the dynamic parameter
- Removed non-standard spaces in prior dictionary accesses (self.priors ["a_alpha"] → self.priors["a_alpha"])

#### model.py
- check_convergence(): fixed if/if/else pattern to if/elif/else
- correct_convergence(): simplified from an if/else return True/False block to a single boolean expression
- get_b_objective_probability(): replaced list(generator) with a list comprehension, consistent with get_a_objective_probability()
- count_belief_a() and count_belief_b(): eliminated Bayesian formula duplication; they now delegate to agent.a_expectations / agent.b_expectations
- DataCollector agent_reporters: replaced lambda a: a.a_expectations() with "a_expectations" (string form, compatible with properties via getattr)
- Bandit.__init__: added self.theory_threshold among model attributes

#### app.py
- deviation_plot: removed dead call ax.get_label() and redundant first ax.legend() (which was immediately overwritten by ax.legend(fontsize=8))
- agents_table: replaced loop string concatenation rows += f"..." with "".join(... for agent in model.agents) — O(n) complexity instead of O(n²)
- agents_table, aggregate_metrics, belief_histogram: updated all calls from agent.a_expectations() / agent.b_expectations() to property syntax (agent.a_expectations / agent.b_expectations)
- generate_results_zip: replaced list(model.agents)[0].inertia and list(model.agents)[0].theory_threshold with model.inertia and model.theory_threshold

**Batch run**
- Now there is only one instance of the model, used both for batch running and for the interactive dashboard: therefore all duplicated code for tests, agent behaviour and model is removed.

**Replications**
- New version of the replication notebooks now run the current model and include more detailed explanations

---

## [0.1.0]

Initial release. Deposited on Zenodo.

### Added

- `Bandit` model with `Scientist` agents on NetworkX graphs (complete, wheel, cycle)
- Bayesian belief updating via Beta distribution accumulation
- Support for Frey & Šešelja (2020) extensions: dynamic objectives, critical interaction, rational inertia, theory threshold
- Interactive Solara dashboard with network visualization, belief trajectory plot, and experimental evidence display
- Batch run script with seaborn plotting and CSV export
- Replication notebooks for Zollman (2010), Rosenstock et al. (2017), and Frey & Šešelja (2020)
- MIT license, README, CONTRIBUTING.md

---

[0.2.0]: https://github.com/sprologuglie/Interactive-Epistemic-ABM-Zollman-Using-Mesa/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sprologuglie/Interactive-Epistemic-ABM-Zollman-Using-Mesa/releases/tag/v0.1.0