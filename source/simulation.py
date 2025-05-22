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

from multiprocessing import Pool, cpu_count

def simulate_process_parallel_processing(df_train, simulation_parameters, data_dir, num_simulations, num_cpus=None):
    start_timestamp = simulation_parameters['case_arrival_times'][0]
    simulation_parameters['start_timestamp'] = start_timestamp
    simulation_parameters['case_arrival_times'] = simulation_parameters['case_arrival_times'][1:]

    args_list = [(scenario_id, df_train, simulation_parameters, data_dir, start_timestamp) for scenario_id in range(num_simulations)
    ]

    tqdm.set_lock(RLock())

    if num_cpus is None:
        num_cpus = cpu_count()

    with Pool(processes=min(num_cpus, num_simulations)) as pool:
        pool.map(simulate_experiment, args_list)


def simulate_experiment(args):
    scenario_id, df_train, simulation_parameters, data_dir, start_timestamp = args
    local_parameters = update_simulation_parameters(simulation_parameters.copy(), scenario_id)
    save_simulation_parameters_for_scenario(local_parameters, data_dir, scenario_id)
    business_process_model = BusinessProcessModel(df_train, local_parameters)
    simulate_scenario(scenario_id, business_process_model, start_timestamp, data_dir, local_parameters)



def simulate_process(df_train, simulation_parameters, data_dir, num_simulations):
    start_timestamp = simulation_parameters['case_arrival_times'][0]
    simulation_parameters['start_timestamp'] = start_timestamp
    simulation_parameters['case_arrival_times'] = simulation_parameters['case_arrival_times'][1:]


    # TODO: count the number of experiment_1 config files

    for scenario_id in range(num_simulations):
        arg_list = scenario_id, df_train, simulation_parameters, data_dir, start_timestamp
        simulate_experiment(arg_list)

def simulate_scenario(scenario_id, business_process_model, start_timestamp, data_dir, simulation_parameters):
    case_id = 0
    case_ = Case(case_id=case_id, start_timestamp=start_timestamp)
    cases = [case_]

    total_cases = len(business_process_model.sampled_case_starting_times)
    progress_bar = tqdm(
        total=total_cases,
        desc=f"Simulation in progress for the scenario {scenario_id + 1}: ",
        #leave=True,
        #position=scenario_id,  # ensures each bar is on its own line
        #dynamic_ncols=True  # adjusts bar width dynamically
        # ascii=True              # optional: use ascii characters for bar
    )

    while business_process_model.sampled_case_starting_times:
        business_process_model.step(cases)
        progress_bar.update(1)

    progress_bar.close()
    print(f"number of simulated cases: {len(business_process_model.past_cases)}")

    simulated_log = pd.DataFrame(business_process_model.simulated_events)
    simulated_log['resource'] = simulated_log['agent'].map(simulation_parameters['agent_to_resource'])
    store_simulated_log(data_dir, simulated_log, scenario_id)
    return None


def update_case_arrivals(simulation_parameters, sim_id, arrival_config):
    start_time = pd.Timestamp(arrival_config["start_time"], tz='UTC')
    end_time = pd.Timestamp(arrival_config["end_time"], tz='UTC')

    new_case_arrival_times = generate_arrivals_case_timestamps_between_times(
        N=arrival_config["N"],
        rate_low=arrival_config["rate_low"],
        rate_high=arrival_config["rate_high"],
        start_time=start_time,
        end_time=end_time,
        num_changes=sim_id // 2 + 1,
        start_with_increase=((sim_id + 1) % 2 == 1)
    )

    simulation_parameters['start_timestamp'] = new_case_arrival_times[0]
    simulation_parameters['case_arrival_times'] = new_case_arrival_times[1:]
    #plot_case_arrival_histogram(new_case_arrival_times, 200)

    return simulation_parameters


def load_scenario_config():
    base_path = Path(__file__).parent.parent # Folder where the script is located
    config_path = base_path /  "raw_data" / "experiment_1_config.yaml"
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

def update_simulation_parameters(simulation_parameters, sim_id):

    scenario_config = load_scenario_config()
    simulation_parameters = update_case_arrivals(simulation_parameters, sim_id, scenario_config["arrivals"])
    simulation_parameters = update_task_duration_dist(simulation_parameters, scenario_config["duration_distribution"])
    #simulation_parameters["activities_without_waiting_time"] = ['zzz_end']
    simulation_parameters = define_agent_availability(simulation_parameters, scenario_config["agent_availability"])

    return simulation_parameters


def define_agent_availability(simulation_parameters, config):
    SS = config["agents_in_SS"]
    num_changes = config["num_changes"]
    resource_funcs = create_individual_availability_functions(SS, num_changes)
    plot_generated_agent_availabilities(resource_funcs)
    simulation_parameters['agent_availability'] = resource_funcs
    return simulation_parameters

def create_individual_availability_functions(SS, num_changes):
    if len(SS) != 2:
        raise ValueError("SS must contain exactly two integers.")

    max_resources = max(SS)
    total_availability_func = create_pattern_function(SS, num_changes)

    resource_functions = {}

    for i in range(1, max_resources + 1):
        # Each resource is available if its index is less than total available at time t
        resource_functions[i] = lambda t, idx=i: 1 if idx <= total_availability_func(t) else 0

    return resource_functions


def plot_generated_agent_availabilities(resource_funcs):
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
    plt.title("Resource Availability Over Time")
    plt.legend(loc="upper right", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def create_pattern_function(SS, num_changes):
    if len(SS) != 2:
        raise ValueError("SS must contain exactly two integers.")

    a, b = SS
    segments = num_changes * 2 + 1
    interval_length = 1 / segments

    def pattern(x):
        if not 0 <= x <= 1:
            raise ValueError("Input x must be between 0 and 1.")

        segment = int(x / interval_length)
        t = (x % interval_length) / interval_length  # normalized position within segment

        if segment % 2 == 0:
            # Constant value segment
            value = a if (segment // 2) % 2 == 0 else b
        else:
            # Linear transition segment
            if a < b:
                # increasing then decreasing
                if (segment // 2) % 2 == 0:
                    value = a + (b - a) * t
                else:
                    value = b - (b - a) * t
            else:
                # decreasing then increasing
                if (segment // 2) % 2 == 0:
                    value = a - (a - b) * t
                else:
                    value = b + (a - b) * t

        return round(value)

    return lambda x: pattern(x)



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


def plot_case_arrival_histogram(timestamps, bins=100):
    plt.figure(figsize=(10, 5))
    plt.hist(timestamps, bins=bins, edgecolor='black')
    plt.xlabel('Time')
    plt.ylabel('Number of Cases')
    plt.title('Histogram of Case Arrivals Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def generate_arrivals_case_timestamps_between_times(N, rate_low, rate_high, num_changes,
                                                    start_time, end_time,
                                                    start_with_increase=True):
    """
    Generate a list of N case arrival timestamps between two given times.

    The arrival rate follows a pattern with a specified number of changes between
    increasing and decreasing linear segments. Between each change, the rate is held
    constant. The total time is split such that the duration spent changing rates equals
    the duration spent at constant rates.

    Parameters:
        N (int): Total number of arrival timestamps to generate.
        rate_low (float): The lower bound of the arrival rate (cases per second).
        rate_high (float): The upper bound of the arrival rate (cases per second).
        num_changes (int): The number of rate changes (alternating increase/decrease).
        start_time (pd.Timestamp): The starting timestamp for the first case.
        end_time (pd.Timestamp): The ending timestamp for the last case.
        start_with_increase (bool): If True, the first change will increase from
            rate_low to rate_high. If False, it will decrease from rate_high to rate_low.

    Returns:
        List[pd.Timestamp]: A list of N pandas Timestamps representing the arrival times
        of the cases, spread between start_time and end_time according to the described pattern.
    """

    # Validate input parameters
    if num_changes < 1:
        raise ValueError("'num_changes' must be at least 1")
    if rate_high <= rate_low:
        raise ValueError("'rate_high' must be greater than 'rate_low'")
    if end_time <= start_time:
        raise ValueError("'end_time' must be after 'start_time'")

    total_duration_seconds = (end_time - start_time).total_seconds()

    change_segments = []
    fixed_segments = []

    current_is_increase = start_with_increase

    ### DEFINE A LIST LAMBDA FUNCTION PER EACH SEGMENT THAT RETURNS THE DESIRED NUMBER OF ARRIVALS PER TIME PERIOD
    # Define the initial constant rate segment based on whether we start with an increase or decrease
    if start_with_increase:
        pattern = [lambda t: rate_low]  # Start flat at low rate
    else:
        pattern = [lambda t: rate_high]  # Start flat at high rate

    # Create alternating increase/decrease and flat segments
    for _ in range(num_changes):
        if current_is_increase:
            change_segments.append(lambda t, low=rate_low, high=rate_high: low + (high - low) * t)
            fixed_segments.append(lambda t: rate_high)
        else:
            change_segments.append(lambda t, low=rate_low, high=rate_high: high - (high - low) * t)
            fixed_segments.append(lambda t: rate_low)
        current_is_increase = not current_is_increase

    # Interleave change and fixed segments into the full rate pattern
    for change, fixed in zip(change_segments, fixed_segments):
        pattern.append(change)
        pattern.append(fixed)

    ### DEFINE A LIST OF DURATIONS IN SECONDS FOR EACH SEGMENT "durations"
    # Compute the number of segments
    total_segments = len(pattern)
    num_change_segments = len(change_segments)
    num_fixed_segments = total_segments - num_change_segments

    # Assign half of the total duration to change segments and the other half to fixed segments (in seconds)
    change_duration = 0.5 * total_duration_seconds / num_change_segments
    fixed_duration = 0.5 * total_duration_seconds / num_fixed_segments

    # Assign durations to each segment based on its type
    durations = [fixed_duration if i % 2 == 0 else change_duration for i in range(total_segments)]

    ### ESTIMATE THE NUMBER OF CASES PER SEGMENT
    # Estimate how many cases should appear in each segment
    segment_cases = []
    total_expected_cases = 0
    for f, d in zip(pattern, durations):
        t = np.linspace(0, 1, 1000)  # Normalized time points within the segment
        r = np.array([f(x) for x in t])  # Rates at each time point
        segment_total = np.trapz(r, t) * d  # Approximate total cases via integration
        #segment_total = np.mean(r)  * d  # Approximate total cases via simple integration
        segment_cases.append(segment_total)
        total_expected_cases += segment_total

    # Scale the number of cases to match the desired total N
    scale = N / total_expected_cases
    segment_cases = [int(round(c * scale)) for c in segment_cases]

    ### GENERATE TIMESTAMPS WITH A GIVEN ARRIVAL RATE
    # Generate timestamps by simulating inter-arrival times
    timestamps = []
    current_offset = 0.0
    for f, duration, count in zip(pattern, durations, segment_cases):
        if count == 0:
            continue
        segment_times = generate_segment_times(f, count, duration)
        segment_times += current_offset  # Shift by current time offset
        current_offset += duration  # Update offset for next segment
        for seconds in segment_times:
            timestamps.append(start_time + pd.to_timedelta(seconds, unit='s'))  # Convert to absolute timestamp

    return timestamps[:N]

def generate_segment_times(rate_fn, count, duration, grid_points=100):
    """
    Generate event times for a non-homogeneous Poisson process segment
    using numerical inversion of the cumulative intensity function.

    This method ensures smooth transitions between different rate regimes
    (e.g., fixed to linear), and avoids discontinuities that occur when
    using simple inverse-rate spacing.

    Parameters:
        rate_fn (callable): A function of a single variable t in [0, 1],
                            defining the normalized rate over the segment.
        count (int): Number of events to generate within the segment.
        duration (float): Length of the segment in seconds.
        grid_points (int): Number of grid points to evaluate the rate and
                           compute the cumulative intensity. Higher values
                           improve accuracy at the cost of performance.

    Returns:
        np.ndarray: A 1D array of event times (in seconds), scaled to fit
                    within the specified duration.
    """
    # Step 1: Define a uniform grid over the interval [0, 1]
    t_grid = np.linspace(0, 1, grid_points)

    # Step 2: Evaluate the rate function on the grid and compute the
    # cumulative intensity using the trapezoidal rule
    rates = np.array([rate_fn(t) for t in t_grid])
    cumulative = cumtrapz(rates, t_grid, initial=0)

    # Step 3: Normalize the cumulative intensity so that it spans [0, 1]
    cumulative /= cumulative[-1]

    # Step 4: Interpolate to construct the inverse of the normalized
    # cumulative intensity function
    inverse_cdf = interp1d(cumulative, t_grid, bounds_error=False, fill_value=(0, 1))

    # Step 5: Map evenly spaced uniform values through the inverse function
    # to obtain event times, then scale them to the segment duration
    uniform_points = np.linspace(0, 1, count)
    local_times = inverse_cdf(uniform_points) * duration

    return local_times


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
        self.resources = sorted(set(self.data['agent']))
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

        for agent_id in range(len(self.resources)):
            # TODO: enable scenario 2
            agent = ResourceAgent(agent_id, self, self.resources[agent_id], self.timer, self.contractor_agent)
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