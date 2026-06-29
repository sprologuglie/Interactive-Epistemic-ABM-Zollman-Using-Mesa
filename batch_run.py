import mesa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from epistemic_abm import Bandit

        
###                 BATCH RUNNER                    ###

parameters_batch_run = {
        "n" : range(5, 12, 1),
        "a_objective" : .5, 
        "b_objective" : .499, 
        "max_priors" : 4,
        "graph" : ["complete", "wheel", "cycle"],
        "step_pulls" : 1000,
        "dynamic" : None,
        "criticism" : None,
        "theory_threshold" : 0,
        "inertia" : 0,
        "seed" : None
    }

batch_run_results = mesa.batch_run(
    Bandit,
    parameters= parameters_batch_run,
    iterations=500,
    max_steps=3000
)

batch_run_results_df = pd.DataFrame(batch_run_results)

sns.lineplot(data=batch_run_results_df, x="n", y="Correct Convergence", hue="graph", errorbar=None)

plt.show()

# For saving to csv

def save_df_csv(df, path=None, *, index=False):
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"outputs/batch_run_{timestamp}.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8")
    return path

save_df_csv(df=batch_run_results_df)