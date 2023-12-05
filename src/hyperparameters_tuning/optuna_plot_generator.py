import pandas as pd
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import datetime


def read_study_from_csv(csv_file: str) -> optuna.study.Study:
    """
    Read Optuna study results from a CSV file and reconstruct a study object.

    :param csv_file: Path to the CSV file containing study results.
    :return: Reconstructed Optuna study object.
    """
    df = pd.read_csv(csv_file)
    study = optuna.create_study(direction="maximize")

    for _, row in df.iterrows():
        # Extract parameters prefixed with 'params_'
        params = {
            key.split("params_")[1]: row[key]
            for key in df.columns
            if key.startswith("params_")
        }
        # Convert strings to datetime
        datetime_start = pd.to_datetime(row["datetime_start"])
        datetime_complete = pd.to_datetime(row["datetime_complete"])
        # Create a trial object
        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                k: optuna.distributions.CategoricalDistribution([v])
                for k, v in params.items()
            },
            value=row["value"],
            intermediate_values={},  # Add intermediate values if available
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(trial)
    return study


def plot_study_results(csv_file: str):
    """
    Generate and display optimization history and parameter importance plots
    from an Optuna study stored in a CSV file.

    :param csv_file: Path to the CSV file containing study results.
    """
    study = read_study_from_csv(csv_file)

    # Generate the plots
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    # Show the plots
    fig1.show()
    fig2.show()


# Path to the CSV file containing the study results
csv_file_path = "study_results_a2c_cartpole.csv"

# Generate and display the plots
plot_study_results(csv_file_path)
