import mesa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from batch_run_scripts.model import Bandit

        
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
    iterations=1,
    max_steps=100
)

batch_run_results_df = pd.DataFrame(batch_run_results)

sns.lineplot(data=batch_run_results_df, x="n", y="Correct Convergence", hue="graph", errorbar=None)

plt.show()

# For saving to csv

def save_df_csv(df, path="outputs/my_dataframe.csv", *, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8")
    return path

# Default path
p = save_df_csv(batch_run_results_df) # To specify you path: save_df_csv(batch_run_results_df, path="my/path/bacth_run_results_df.csv")   
print("Salvato in:", p)
