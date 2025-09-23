import pandas as pd
import yaml
import pandas as pd
from pathlib import Path
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from source.agents.contractor import ContractorAgent
from source.agents.resource import ResourceAgent
from source.utils import store_simulated_log
import matplotlib.pyplot as plt
from source.arrival_distribution import DurationDistribution
from source.utils import save_simulation_parameters_for_scenario
from tqdm import tqdm
from threading import RLock
import re
from multiprocessing import Pool, cpu_count
import os

def simulate_process_parallel_processing(df_train, simulation_parameters, data_dir, num_simulations, num_cpus=None):
    start_timestamp = simulation_parameters['case_arrival_times'][0]
    simulation_parameters['start_timestamp'] = start_timestamp
    simulation_parameters['case_arrival_times'] = simulation_parameters['case_arrival_times'][1:]

    num_of_scenarios = count_experiment_configs(data_dir)

    args_list = [(scenario_id, scenario_run_id, df_train, simulation_parameters, data_dir, start_timestamp)
        for scenario_id in range(1, num_of_scenarios + 1)
        for scenario_run_id in range(1, num_simulations + 1)]

    tqdm.set_lock(RLock())

    if num_cpus is None:
        num_cpus = cpu_count()

    with Pool(processes=min(num_cpus, num_of_scenarios)) as pool:
        pool.map(simulate_experiment, args_list)


def simulate_experiment(args):
    scenario_id, scenario_run_id, df_train, simulation_parameters, data_dir, start_timestamp = args
    simulation_parameters_updated = update_simulation_parameters(simulation_parameters.copy(), scenario_id)
    business_process_model = BusinessProcessModel(df_train, simulation_parameters_updated)
    simulate_scenario(scenario_id, scenario_run_id, business_process_model, start_timestamp, data_dir, simulation_parameters_updated)



def simulate_process(df_train, simulation_parameters, data_dir, num_simulations):
    start_timestamp = simulation_parameters['case_arrival_times'][0]
    simulation_parameters['start_timestamp'] = start_timestamp
    simulation_parameters['case_arrival_times'] = simulation_parameters['case_arrival_times'][1:]

    num_of_scenarios = count_experiment_configs(data_dir)

    args_list = [(scenario_id, scenario_run_id, df_train, simulation_parameters, data_dir, start_timestamp)
        for scenario_id in range(1, num_of_scenarios + 1)
        for scenario_run_id in range(1, num_simulations + 1)]

    #for args in [1]:
    for args in args_list:
        simulate_experiment(args)

def simulate_scenario(scenario_id, scenario_run_id, business_process_model, start_timestamp, data_dir, simulation_parameters):
    case_id = 0
    case_ = Case(case_id=case_id, start_timestamp=start_timestamp)
    cases = [case_]

    total_cases = len(business_process_model.sampled_case_starting_times)
    progress_bar = tqdm(
        total=total_cases,
        desc=f"Scenario {scenario_id}, simulation run {scenario_run_id}",
    )

    while business_process_model.sampled_case_starting_times:
        business_process_model.step(cases)
        progress_bar.update(1)

    progress_bar.close()
    print(f"Number of simulated cases: {len(business_process_model.past_cases)}")

    simulated_log = pd.DataFrame(business_process_model.simulated_events)
    simulated_log['resource'] = simulated_log['agent'].map(simulation_parameters['agent_to_resource'])

    simulated_log = add_ss_id_arrivals(simulated_log, simulation_parameters)
    simulated_log = add_ss_id_resources(simulated_log, simulation_parameters)

    store_simulated_log(data_dir, simulated_log, scenario_id, scenario_run_id)

    return None


def add_ss_id_resources(simulated_log, simulation_parameters):
    gold_standard = simulation_parameters['gold_standard']['scenario_resources']
    simulated_log['ss_id_resources'] = simulated_log["start_timestamp"].apply(
        lambda ts: get_stead_state_id(ts, gold_standard))
    return simulated_log


def add_ss_id_arrivals(simulated_log, simulation_parameters):
    gold_standard = simulation_parameters['gold_standard']['scenario_arrivals']

    simulated_log['ss_id_arrivals'] = simulated_log["start_timestamp"].apply(
        lambda ts: get_stead_state_id(ts, gold_standard))
    return simulated_log



def get_stead_state_id(timestamp, gold_standard):
    """
    Given a timestamp, return the segment_id and ss_id from segment metadata.

    Parameters:
        timestamp (pd.Timestamp): The timestamp to classify.
        gold_standard (List[dict]): List of segment information dictionaries,
                                       each with keys 'start', 'end', 'segment_id', and 'ss_id'.

    Returns:
        dict: A dictionary with keys 'segment_id' and 'ss_id'.
              Returns None if timestamp does not fall in any segment.
    """
    for segment in gold_standard:
        if segment["start"] <= timestamp < segment["end"]:
            return int(segment["ss_id"])
    return -1


def count_experiment_configs(data_dir):
    """
    Count files named experiment_<number>_config.yaml in the given folder.

    Parameters:
        folder_path (str): Path to the directory to search.

    Returns:
        int: Number of files matching the pattern.
    """

    folder_path = change_data_dir_to_folder_with_config(data_dir)

    pattern = re.compile(r'^experiment_1_config_\d+\.yaml$')
    try:
        entries = os.listdir(folder_path)
    except FileNotFoundError:
        raise ValueError(f"Folder not found: {folder_path}")

    count = sum(1 for name in entries if pattern.match(name))
    print(f"Found {count} configurations files for Experiment 1")
    return count



def add_warm_up_arrivals_v2(case_arrival_times, first_ss_metadata, log_ratio=0.1):
    """
    Adds warm-up arrival timestamps by prepending N synthetic arrivals,
    generated using the average interarrival rate from the first segment.

    Parameters:
        case_arrival_times (list): List of pandas Timestamps (must be timezone-aware).
        first_ss_metadata (dict): Dict with 'start' and 'end' Timestamps of the first segment.
        SS_ratio (float): Ratio of first-segment timestamps to base warm-up count on (default: 0.5).
        log_ratio (float): Minimum ratio of total timestamps to use for warm-up count (default: 0.1).

    Returns:
        list: Combined list of warm-up and original timestamps, sorted chronologically.
    """

    # Step 1: Convert and sort the original timestamps
    timestamps = pd.Series(case_arrival_times).sort_values()

    # Step 2: Get time window from metadata
    start = first_ss_metadata['start']
    end = first_ss_metadata['end']

    # Step 3: Filter timestamps within the first segment
    filtered = timestamps[(timestamps >= start) & (timestamps <= end)].sort_values()

    # Step 4: Determine how many new warm-up arrivals to generate
    N =  int(len(case_arrival_times) * log_ratio)

    # Step 5: Compute average interarrival time
    avg_interarrival = (filtered.iloc[-1] - filtered.iloc[0]) / max(len(filtered) - 1, 1)

    # Step 6: Generate N new timestamps before the earliest original timestamp
    earliest = timestamps.iloc[0]
    prepended = [earliest - i * avg_interarrival for i in range(N, 0, -1)]  # in order

    # Step 7: Combine and return sorted list
    combined = pd.Series(prepended + timestamps.tolist()).sort_values().reset_index(drop=True)
    return combined.tolist()


def add_warm_up_arrivals(case_arrival_times, first_ss_metadata, SS_ratio=0.5, log_ratio=0.1):
    """
    Adds warm-up arrival timestamps by prepending a shifted subset of early arrivals.

    Parameters:
        case_arrival_times (list): List of pandas Timestamps (must be timezone-aware).
        first_ss_metadata (list): List of dicts with 'start' and 'end' Timestamps.
        SS_ratio (float): Ratio of filtered timestamps to use for the warm-up subset (default: 0.5).
        log_ratio (float): Minimum ratio of total timestamps to include (default: 0.1).

    Returns:
        list: Combined list of warm-up and original timestamps, sorted chronologically.
    """

    # Step 1: Convert and sort the original timestamps
    timestamps = pd.Series(case_arrival_times).sort_values()

    # Step 2: Get time window from segment metadata
    start = first_ss_metadata['start']
    end = first_ss_metadata['end']

    # Step 3: Filter timestamps within the segment window
    filtered = timestamps[(timestamps >= start) & (timestamps <= end)]

    # Step 4: Use the larger of SS_ratio (filtered) or log_ratio (global)
    subset_size = max(int(len(filtered) * SS_ratio), int(len(case_arrival_times) * log_ratio))
    subset = filtered.iloc[:subset_size]

    # Step 5: Shift the subset backward by the time from global start to end of subset
    shift_duration = subset.iloc[-1] - timestamps.iloc[0]
    shifted = subset - shift_duration

    # Step 6: Combine (excluding last shifted to avoid overlap) and sort
    combined = pd.concat([shifted[:-1], timestamps]).sort_values().reset_index(drop=True)

    return combined.tolist()


def update_case_arrivals(simulation_parameters, scenario_id, arrival_config):

    case_arrival_times, segment_metadata = generate_arrivals_case_timestamps_between_times_new(
        N=arrival_config["N"],
        rate_schedule=arrival_config["rate_schedule"],
        start_time=pd.Timestamp(arrival_config["start_time"], tz='UTC'),
        end_time=pd.Timestamp(arrival_config["end_time"], tz='UTC'))

    #case_arrival_times = add_warm_up_arrivals(case_arrival_times, segment_metadata[0])
    case_arrival_times = add_warm_up_arrivals_v2(case_arrival_times, segment_metadata[0])

    if simulation_parameters['plot_on']:
        plot_case_arrival_histogram(case_arrival_times, scenario_id, 200)

    simulation_parameters['start_timestamp'] = case_arrival_times[0]
    simulation_parameters['case_arrival_times'] = case_arrival_times[1:]
    simulation_parameters['gold_standard'] = {'scenario_arrivals': segment_metadata}

    return simulation_parameters


def load_scenario_config(sim_id):
    base_path = Path(__file__).parent.parent # Folder where the script is located
    config_path = base_path /  "raw_data" / "experiment_1_settings" / f"experiment_1_config_{sim_id}.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def update_task_duration_dist(simulation_parameters, duration_config):
    dist_type = duration_config["type"]
    mean = duration_config.get("mean")
    std = duration_config.get("std")
    shape = duration_config.get("shape")
    minimum = duration_config.get("min")
    maximum = duration_config.get("max")

    new_dist = DurationDistribution(dist_type, mean, std, shape, minimum, maximum)
    updated_dist = replace_distributions(simulation_parameters['activity_durations_dict'], new_dist)
    simulation_parameters['activity_durations_dict'] = updated_dist
    return simulation_parameters

def update_simulation_parameters(simulation_parameters, scenario_id):

    scenario_config = load_scenario_config(scenario_id)
    simulation_parameters = update_case_arrivals(simulation_parameters, scenario_id, scenario_config["arrivals"])
    simulation_parameters = update_task_duration_dist(simulation_parameters, scenario_config["duration_distribution"])
    #simulation_parameters["activities_without_waiting_time"] = ['zzz_end']
    simulation_parameters = define_agent_availability(simulation_parameters, scenario_config["agent_availability"], scenario_id)
    simulation_parameters = update_resource_related_config(simulation_parameters)
    return simulation_parameters


def update_role(simulation_parameters, n_agents):

    agents =  list(range(n_agents))
    calendar = [{'from': 'MONDAY',
      'to': 'MONDAY',
      'beginTime': '00:00:00',
      'endTime': '23:59:59.999000'},
     {'from': 'TUESDAY',
      'to': 'TUESDAY',
      'beginTime': '00:00:00',
      'endTime': '23:59:59.999000'},
     {'from': 'WEDNESDAY',
      'to': 'WEDNESDAY',
      'beginTime': '00:00:00',
      'endTime': '23:59:59.999000'},
     {'from': 'THURSDAY',
      'to': 'THURSDAY',
      'beginTime': '00:00:00',
      'endTime': '23:59:59.999000'},
     {'from': 'FRIDAY',
      'to': 'FRIDAY',
      'beginTime': '00:00:00',
      'endTime': '23:59:59.999000'},
     {'from': 'SATURDAY',
      'to': 'SATURDAY',
      'beginTime': '00:00:00',
      'endTime': '23:59:59.999000'},
     {'from': 'SUNDAY',
      'to': 'SUNDAY',
      'beginTime': '00:00:00',
      'endTime': '23:59:59.999000'}]

    roles = {'Role 1': {'agents': agents, 'calendar': calendar}}
    simulation_parameters['roles'] = roles
    return simulation_parameters


def update_resource_related_config(simulation_parameters):

    # Determine the number of agents originally detected from the bimp simulated event log
    number_of_original_agents = len(simulation_parameters['res_calendars'])

    # Determine the number of relevant agents from the 'agent_availability' dictionary
    number_of_relevant_agents = len(simulation_parameters["agent_availability"].keys())

    # Update the 'Role 1' section of simulation_parameters based on the number of agents
    simulation_parameters = update_role(simulation_parameters, number_of_relevant_agents)

    # Define the list of keys that require updating
    key_to_be_updated = [
        "activity_durations_dict",
        "res_calendars",
        "agent_activity_mapping",
        "agent_transition_probabilities_autonomous",
        "agent_to_resource"
    ]

    # For each relevant key, update the values to match the current number of agents
    for key in key_to_be_updated:
        if key in simulation_parameters:
            updated_values = {}
            for agent_id in range(number_of_relevant_agents):
                # Use modulo to safely get a base value even if original keys are fewer
                base_value = simulation_parameters[key].get(agent_id % number_of_original_agents)
                updated_values[agent_id] = base_value
            # Replace the original dictionary with the updated one
            simulation_parameters[key] = updated_values

    simulation_parameters['agents_sorted'] =  range(number_of_relevant_agents)

    return simulation_parameters


def define_agent_availability(simulation_parameters, config, scenario_id):
    agents_in_SS = config["agents_in_SS"]
    resource_funcs = create_individual_availability_functions(agents_in_SS)
    simulation_parameters['agent_availability'] = resource_funcs

    if simulation_parameters['plot_on']:
        plot_generated_agent_availabilities(resource_funcs, scenario_id)

    gold_standard_resources = create_metadata_resources(agents_in_SS,
                                     simulation_parameters['start_timestamp'],
                                     simulation_parameters['case_arrival_times'][-1])
    simulation_parameters['gold_standard']['scenario_resources'] = gold_standard_resources

    return simulation_parameters


def create_individual_availability_functions(agents_in_SS):

    max_resources = max(agents_in_SS)
    total_availability_func = create_pattern_function(agents_in_SS)

    resource_functions = {}
    for i in range(1, max_resources + 1):
        # Each resource is available if its index is less than total available at time t
        # Selects the first N resources to be employed, where N is the required number of resources
        resource_functions[i] = lambda t, idx=i: 1 if idx <= total_availability_func(t) else 0

    return resource_functions


def plot_generated_agent_availabilities(resource_funcs, scenario_id):
    # Plotting
    t_values = np.linspace(0, 1, 1000)

    # Calculate total availability over time
    total_availability = [sum(func(t) for func in resource_funcs.values()) for t in t_values]

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot individual resource availability
    for i, func in resource_funcs.items():
        y_values = [func(t) for t in t_values]
        plt.plot(t_values, y_values, label=f"Resource {i}", alpha=0.6)

    # Plot total availability
    plt.plot(t_values, total_availability, label="Total Availability", color='black', linewidth=2)

    plt.xlabel("Time (t)")
    plt.ylabel("Availability (0 or 1)")
    plt.title(f"Resource Availability Over Time (scenario {scenario_id})")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def create_metadata_resources(agents_in_SS, start_point, end_point):
    """
    Create metadata from agent steady-state information to be used as a gold standard in evaluation.

    Parameters:
        agents_in_SS (list[int]): Sequence of resource levels at fixed segments.
        start_point (pd.Timestamp): Start time of the interval.
        end_point (pd.Timestamp): End time of the interval.

    Returns:
        list[dict]: Metadata for each segment, including fix and change segments.
    """
    metadata = []
    segment_id = 1  # Unique segment identifier starting at 1
    length = len(agents_in_SS)  # Total number of fixed resource values

    # Calculate total duration in seconds and duration per segment step
    total_duration = (end_point - start_point).total_seconds()
    seconds_per_step = total_duration / length

    ss_id_map = {}  # Map to assign unique ss_id for each resource value
    current_ss_id = 1  # Counter for ss_id assignments

    for i in range(length):
        resource_value = agents_in_SS[i]

        # Assign ss_id for the fixed segment if not already assigned
        if resource_value not in ss_id_map:
            ss_id_map[resource_value] = current_ss_id
            current_ss_id += 1
        fix_ss_id = ss_id_map[resource_value]

        # Define start time for the fix segment
        start_fix = start_point + pd.to_timedelta(i * seconds_per_step, unit='s')

        # Define end time for fix segment:
        # For all but the last fixed segment, the fix segment ends halfway to the next point
        if i < length - 1:
            end_fix = start_point + pd.to_timedelta((i + 0.5) * seconds_per_step, unit='s')
        else:
            # For the last fixed segment, end at the final end_point
            end_fix = start_point + pd.to_timedelta((i + 1) * seconds_per_step, unit='s')

        # Append fix segment metadata
        metadata.append({
            "start": start_fix,
            "end": end_fix,
            "type": "fix",
            "segment_id": segment_id,
            "num_resources": resource_value,
            "ss_id": fix_ss_id
        })
        segment_id += 1  # Increment segment id for next segment

        # Add a change segment only if this is not the last fixed segment
        if i < length - 1:
            next_value = agents_in_SS[i + 1]

            # Determine if the change is an increase or decrease
            change_type = "increase" if next_value > resource_value else "decrease"

            # Change segment starts where the fix segment ends
            start_change = end_fix

            # Change segment ends at the start of the next fix segment
            end_change = start_point + pd.to_timedelta((i + 1) * seconds_per_step, unit='s')

            # Append change segment metadata with no fixed resource count and ss_id = 0
            metadata.append({
                "start": start_change,
                "end": end_change,
                "type": change_type,
                "segment_id": segment_id,
                "num_resources": float('nan'),
                "ss_id": 0
            })
            segment_id += 1  # Increment segment id for next segment

    return metadata



def create_pattern_function(agents_in_SS):
    """
    Build a piecewise pattern function that stays constant at each
    value in agents_in_SS, with linear transitions between them.

    Parameters:
        agents_in_SS (list of int): Steady‐state values, at least one.

    Returns:
        function: A function f(t) mapping t in [0,1] to one of the
                  steady‐state values, with linear ramps in between.
    """
    if not agents_in_SS:
        raise ValueError("agents_in_SS must contain at least one value.")

    n = len(agents_in_SS)
    # Number of segments: constant + transition + constant + … = 2*n - 1
    segments = 2 * n - 1
    interval_length = 1 / segments

    def pattern(t):
        if not 0 <= t <= 1:
            raise ValueError("Input t must be between 0 and 1.")

        # Determine segment index (0 to segments-1)
        seg = min(int(t / interval_length), segments - 1)
        # Position within segment, normalized [0,1)
        tau = (t - seg * interval_length) / interval_length

        if seg % 2 == 0:
            # Even segments are constant
            idx = seg // 2
            value = agents_in_SS[idx]
        else:
            # Odd segments are transitions between agents_in_SS[k] → agents_in_SS[k+1]
            k = seg // 2
            a = agents_in_SS[k]
            b = agents_in_SS[k + 1]
            value = a + (b - a) * tau

        return round(value)

    return pattern


def replace_distributions(distribution_dict, new_distribution):
    """
    Replaces all values in a nested dictionary structure with the given distribution.

    Parameters:
    - distribution_dict (dict): Dictionary of the form {int: {str: DurationDistribution}}
    - new_distribution (object): New distribution object to assign to all keys

    Returns:
    - dict: Modified dictionary with updated distribution objects
    """
    for outer_key in distribution_dict:
        inner_dict = distribution_dict[outer_key]
        for inner_key in inner_dict:
            inner_dict[inner_key] = new_distribution
    return distribution_dict


def plot_case_arrival_histogram(timestamps, scenario_id, bins=100):
    plt.figure(figsize=(10, 5))
    plt.hist(timestamps, bins=bins, edgecolor='black')
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title(f'Histogram of Case Arrivals Over Time (scenario {scenario_id})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_segment_times_new(rate_function, count, duration):
    """
     Generate `count` arrival times over `duration` seconds given a rate_function.
     The function defines rate(t), where t ∈ [0, 1].
     """
    if count == 0:
        return np.array([])

    t_values = np.linspace(0, 1, 1000)
    rate_values = np.array([rate_function(t) for t in t_values], dtype=np.float64)
    cumulative = np.cumsum(rate_values)
    cumulative = cumulative / cumulative[-1]  # Avoid in-place division to preserve dtype
    arrival_points = np.linspace(0, 1, count)
    arrival_times = np.interp(arrival_points, cumulative, t_values)
    return arrival_times * duration

def generate_arrivals_case_timestamps_between_times_new(N, rate_schedule,
                                                    start_time, end_time):
    """
    Generate a list of N case arrival timestamps between two given times.

    The arrival rate follows a pattern defined by the rate_schedule. Each fixed segment
    uses a constant rate from the schedule, and each transition segment linearly changes
    from one rate to the next. The total time is split evenly between constant and changing
    segments.

    Parameters:
        N (int): Total number of arrival timestamps to generate.
        rate_schedule (List[float]): List of target rates (cases per second) for fixed segments.
        start_time (pd.Timestamp): The starting timestamp for the first case.
        end_time (pd.Timestamp): The ending timestamp for the last case.

    Returns:
        List[pd.Timestamp]: A list of N pandas Timestamps representing the arrival times.
    """

    if len(rate_schedule) < 1:
        raise ValueError("'rate_schedule' must have at least one value")
    if end_time <= start_time:
        raise ValueError("'end_time' must be after 'start_time'")

    if len(rate_schedule) == 1:
        if N == 0:
            return []
        total_seconds = (end_time - start_time).total_seconds()
        timestamps = [
            start_time + pd.to_timedelta(i * total_seconds / (N - 1), unit='s')
            for i in range(N)
        ] if N > 1 else [start_time]

        # Create segment metadata for the single fixed segment
        segment_metadata = [{
            "start": start_time,
            "end": end_time,
            "type": "fixed",
            "segment_id": 1,
            "ss_id": 1
        }]
        timestamps = pd.Series(timestamps).sort_values().to_list()
        return timestamps[:N], segment_metadata

    total_duration_seconds = (end_time - start_time).total_seconds()
    num_fixed_segments = len(rate_schedule)
    num_change_segments = len(rate_schedule) - 1

    # Build fixed rate functions
    fixed_segments = [
        (lambda rate: lambda t: rate)(rate) for rate in rate_schedule
    ]

    # Build change rate functions (linear interpolation between rates)
    change_segments = [
        (lambda r1, r2: lambda t: r1 + (r2 - r1) * t)(rate_schedule[i], rate_schedule[i + 1])
        for i in range(num_change_segments)
    ]

    # Interleave fixed and change segments
    pattern = []
    for i in range(num_change_segments):
        pattern.append(fixed_segments[i])
        pattern.append(change_segments[i])
    pattern.append(fixed_segments[-1])  # Final fixed segment

    total_segments = len(pattern)

    # Assign durations: half for fixed, half for changes
    fixed_duration = 0.5 * total_duration_seconds / num_fixed_segments
    change_duration = 0.5 * total_duration_seconds / num_change_segments
    durations = []
    for i in range(total_segments):
        if i % 2 == 0:
            durations.append(fixed_duration)
        else:
            durations.append(change_duration)

    # Estimate cases per segment
    segment_cases = []
    total_expected_cases = 0
    for f, d in zip(pattern, durations):
        t = np.linspace(0, 1, 1000)
        r = np.array([f(x) for x in t])
        segment_total = np.trapz(r, t) * d
        segment_cases.append(segment_total)
        total_expected_cases += segment_total

    # Scale to match total N
    scale = N / total_expected_cases
    segment_cases = [int(round(c * scale)) for c in segment_cases]

    # Generate timestamps
    timestamps = []
    current_offset = 0.0
    for f, duration, count in zip(pattern, durations, segment_cases):
        if count == 0:
            continue
        segment_times = generate_segment_times_new(f, count, duration)
        segment_times += current_offset
        current_offset += duration
        for seconds in segment_times:
            timestamps.append(start_time + pd.to_timedelta(seconds, unit='s'))


    # Create gold standard (segment metadata)
    segment_metadata = create_segment_meda_data(rate_schedule, durations, pattern, start_time)

    timestamps = pd.Series(timestamps).sort_values().to_list()

    return timestamps[:N], segment_metadata


def create_segment_meda_data(rate_schedule, durations, pattern, start_time):

    # Collect segment metadata for gold standard
    # ss_id: fixed segments with the same rate have the same ID; change segments get ss_id = 0
    # segment_id: unique ID per segment (fixed or change), based on pattern order
    segment_metadata = []
    current_offset = 0.0

    # Map each unique fixed rate to a steady-state ID (ss_id)
    rate_to_ss_id = {}
    current_ss_id = 1
    for rate in rate_schedule:
        if rate not in rate_to_ss_id:
            rate_to_ss_id[rate] = current_ss_id
            current_ss_id += 1

    for i, (duration, f) in enumerate(zip(durations, pattern)):
        start = start_time + pd.to_timedelta(current_offset, unit='s')
        end = start_time + pd.to_timedelta(current_offset + duration, unit='s')
        segment_type = "fixed" if i % 2 == 0 else "change"
        if segment_type == "fixed":
            rate = rate_schedule[i // 2]
            ss_id = rate_to_ss_id[rate]
        else:
            ss_id = 0  # For changing segments

        segment_metadata.append({
            "start": start,
            "end": end,
            "type": segment_type,
            "segment_id": i,  # Unique ID for each segment
            "ss_id": ss_id    # Same ID for segments with same rate, 0 for change
        })
        current_offset += duration
    return segment_metadata


class Case:
    """
    represents a case, for example a patient in a medical surveillance process
    """
    def __init__(self, case_id, start_timestamp=None, ) -> None:
        self.case_id = case_id
        self.is_done = False
        self.activities_performed = []
        self.case_start_timestamp = start_timestamp
        self.current_timestamp = start_timestamp
        self.additional_next_activities = []
        self.potential_additional_agents = []
        self.timestamp_before_and_gateway = start_timestamp
        self.previous_agent = -1

    def get_last_activity(self):
        """
        get last activity that happened in the current case
        """
        if len(self.activities_performed) == 0:
            return None
        else:
            return self.activities_performed[-1]
        
    def add_activity_to_case(self, activity):
        self.activities_performed.append(activity)
    
    def update_current_timestep(self, duration):
        self.current_timestamp += pd.Timedelta(seconds=duration)


class BusinessProcessModel(Model):
    def __init__(self, data, simulation_parameters):
        self.data = data
        #self.resources = sorted(set(self.data['agent']))
        self.resources = simulation_parameters['agents_sorted']
        activities = sorted(set(self.data['activity_name']))
        self.roles = simulation_parameters['roles']
        self.agents_busy_until = {key: simulation_parameters['start_timestamp'] for key in self.resources}
        self.calendars = simulation_parameters['res_calendars']
        self.activity_durations_dict = simulation_parameters['activity_durations_dict']
        self.sampled_case_starting_times = simulation_parameters['case_arrival_times']
        self.past_cases = []
        self.maximum_case_id = 0
        self.prerequisites = simulation_parameters['prerequisites']
        self.max_activity_count_per_case = simulation_parameters['max_activity_count_per_case']
        self.timer = simulation_parameters['timers']
        self.activities_without_waiting_time = simulation_parameters['activities_without_waiting_time']
        self.agent_transition_probabilities = simulation_parameters['agent_transition_probabilities']
        self.central_orchestration = simulation_parameters['central_orchestration']
        self.discover_parallel_work = False
        self.schedule = MyScheduler(self,)
        self.contractor_agent = ContractorAgent(unique_id=9999, 
                                                model=self, 
                                                activities=activities, 
                                                transition_probabilities=simulation_parameters['transition_probabilities'], 
                                                agent_activity_mapping=simulation_parameters['agent_activity_mapping'])
        self.schedule.add(self.contractor_agent)
        self.agent_availability = simulation_parameters["agent_availability"]

        for agent_id in range(len(self.resources)):
            agent = ResourceAgent(agent_id, self, self.resources[agent_id], self.timer,
                                  self.agent_availability[agent_id+1],
                                  simulation_parameters["start_timestamp"],
                                  simulation_parameters["case_arrival_times"][-1],
                                  self.contractor_agent)
            self.schedule.add(agent)

        # Data collector to track agent activities over time
        self.datacollector = DataCollector(agent_reporters={"Activity": "current_activity_index"})
        self.simulated_events = []


    def step(self, cases):
        # check if there are still cases planned to arrive in the future
        if len(self.sampled_case_starting_times) > 1:
        # if there are still cases happening
            if cases:
                last_case = cases[-1]
                if last_case.current_timestamp >= self.sampled_case_starting_times[0]:
                    self.maximum_case_id += 1
                    new_case_id = self.maximum_case_id
                    new_case = Case(case_id=new_case_id, start_timestamp=self.sampled_case_starting_times[0])
                    cases.append(new_case)
                    # remove added case from sampled_case_starting_times list
                    self.sampled_case_starting_times = self.sampled_case_starting_times[1:]
            # if no cases are happening
            else:
                self.maximum_case_id += 1
                new_case_id = self.maximum_case_id
                new_case = Case(case_id=new_case_id, start_timestamp=self.sampled_case_starting_times[0])
                cases.append(new_case)
                # remove added case from sampled_case_starting_times list
                self.sampled_case_starting_times = self.sampled_case_starting_times[1:]
        # Sort cases by current timestamp
        cases.sort(key=lambda x: x.current_timestamp)
        # print(f"cases after sorting: {[case.current_timestamp for case in cases]}")
        # print("NEW SIMULATION STEP")
        for case in cases:
            current_active_agents, case_ended = self.contractor_agent.get_potential_agents(case=case)
            if case_ended:
                self.past_cases.append(case)
                cases.remove(case)
                if len(self.sampled_case_starting_times) == 1 and len(cases) == 0:
                    self.sampled_case_starting_times = self.sampled_case_starting_times[1:]
                continue
            if current_active_agents == None:
                continue # continue with next case
            else:
                current_active_agents_sampled = current_active_agents
                self.schedule.step(cases=cases, current_active_agents=current_active_agents_sampled)



class MyScheduler(BaseScheduler):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def step(self, cases, current_active_agents=None):
        """
        Step through the agents, activating each agent in a dynamic subset.
        """
        self.do_each(method="step", agent_keys=current_active_agents, cases=cases)
        self.steps += 1
        self.time += 1

    def get_agent_count(self):
        """
        Returns the current number of active agents in the model.
        """
        return len(self._agents)
    
    def do_each(self, method, cases, agent_keys=None, shuffle=False):
        agent_keys_ = [agent_keys[0]] 
        if agent_keys_ is None:
            agent_keys_ = self.get_agent_keys()
        if shuffle:
            self.model.random.shuffle(agent_keys_)
        for agent_key in agent_keys_:
            if agent_key in self._agents:
                getattr(self._agents[agent_key], method)(self, agent_keys, cases)


def change_data_dir_to_folder_with_config(data_dir):
    base_path = Path(data_dir).parents[3]
    return base_path / 'AgentSimulator' / 'raw_data' / 'experiment_1_settings'


def change_data_dir_to_folder_with_original_model_config(data_dir):
    base_path = Path(data_dir).parents[3]
    return base_path / 'AgentSimulator' / 'raw_data' / 'experiment_1_settings' / 'original_model' / 'simulation_parameters_original_bimp.pkl'
