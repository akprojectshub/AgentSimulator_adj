import os
import math
import pickle
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

import scipy.stats as st

def store_preprocessed_data(df_train, df_test, df_val, data_dir):
    print(data_dir)
    os.system(f"mkdir {data_dir}")
    if not os.path.exists(data_dir):
    # If it doesn't exist, create the directory
        os.makedirs(data_dir)

    path_to_train_file = os.path.join(data_dir,"train_preprocessed.csv")
    df_train_without_end_activity = df_train.copy()
    df_train_without_end_activity = df_train_without_end_activity[df_train_without_end_activity['activity_name'] != 'zzz_end']
    #df_train_without_end_activity.to_csv(path_to_train_file, index=False)

    # save test data
    path_to_test_file = os.path.join(data_dir,"test_preprocessed.csv")
    df_test_without_end_activity = df_test.copy()
    df_test_without_end_activity = df_test_without_end_activity[df_test_without_end_activity['activity_name'] != 'zzz_end']
    #df_test_without_end_activity.to_csv(path_to_test_file, index=False)

    return df_train_without_end_activity


def store_simulated_log(data_dir, simulated_log, scenario_id, simulation_id):
    path_to_file = os.path.join(data_dir,f"simulated_log_{scenario_id}_{simulation_id}.csv")
    simulated_log.to_csv(path_to_file, index=False)

    renamed_log = simulated_log.rename(columns={
        'case_id': 'case:concept:name',
        'activity_name': 'concept:name',
        'end_timestamp': 'time:timestamp',
        'agent': 'org:resource'
    })
    renamed_log = dataframe_utils.convert_timestamp_columns_in_df(renamed_log)
    renamed_log_xes = log_converter.apply(renamed_log, variant=log_converter.Variants.TO_EVENT_LOG)
    path_to_file = os.path.join(data_dir,f"simulated_log_{scenario_id}_{simulation_id}.xes")
    xes_exporter.apply(renamed_log_xes, path_to_file)
    print(f"Simulated logs are stored in {path_to_file}")


def sample_from_distribution(distribution):
    if distribution.type.value == "expon":
        scale = distribution.mean - distribution.min
        if scale < 0.0:
            print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
            scale = distribution.mean
        sample = st.expon.rvs(loc=distribution.min, scale=scale, size=1)
    elif distribution.type.value == "gamma":
        # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        sample = st.gamma.rvs(
            pow(distribution.mean, 2) / distribution.var,
            loc=0,
            scale=distribution.var / distribution.mean,
            size=1,
        )
    elif distribution.type.value == "norm":
        sample = st.norm.rvs(loc=distribution.mean, scale=distribution.std, size=1)
    elif distribution.type.value == "uniform":
        sample = st.uniform.rvs(loc=distribution.min, scale=distribution.max - distribution.min, size=1)
    elif distribution.type.value == "lognorm":
        # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        pow_mean = pow(distribution.mean, 2)
        phi = math.sqrt(distribution.var + pow_mean)
        mu = math.log(pow_mean / phi)
        sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
        sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)
    elif distribution.type.value == "fix":
        sample = [distribution.mean] * 1

    return sample[0]


def save_simulation_parameters_original(simulation_parameters, data_dir, option='pkl'):
    """Save the simulation_parameters dictionary to a text and pickle file in the specified directory."""
    os.makedirs(data_dir, exist_ok=True)

    if option=="pkl":
        pkl_path = os.path.join(data_dir, "", "simulation_parameters_original_bimp.pkl")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(simulation_parameters, pkl_file)
    else:
        txt_path = os.path.join(data_dir, "simulation_parameters_original_bimp.txt")

        with open(txt_path, "w") as file:
            for key, value in simulation_parameters.items():
                file.write(f"{key}: {value}\n")


def save_simulation_parameters_for_scenario(simulation_parameters, data_dir, scenario_id, option='pkl'):
    """Save the simulation_parameters dictionary to a text and pickle file in the specified directory."""
    os.makedirs(data_dir, exist_ok=True)

    if option=="pkl":
        pkl_path = os.path.join(data_dir, f"simulation_parameters_scenario_{scenario_id}.pkl")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(simulation_parameters, pkl_file)
    else:
        txt_path = os.path.join(data_dir, f"simulation_parameters_scenario_{scenario_id}.txt")

        with open(txt_path, "w") as file:
            for key, value in simulation_parameters.items():
                file.write(f"{key}: {value}\n")
