import argparse
import warnings
import time

from source.agent_simulator import AgentSimulator
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process event log parameters')
    
    # File paths
    parser.add_argument('--log_path', help='Path to single log file, which is split into train and test')
    parser.add_argument('--train_path', help='Path to training log file')
    parser.add_argument('--test_path', help='Path to test log file')
    
    # Column names
    parser.add_argument('--case_id', help='Case ID column name')
    parser.add_argument('--activity_name', help='Activity name column')
    parser.add_argument('--resource_name', help='Resource column name')
    parser.add_argument('--end_timestamp', help='End timestamp column name')
    parser.add_argument('--start_timestamp', help='Start timestamp column name')
    
    # Hyperparameters
    parser.add_argument('--extr_delays', action='store_true', help='Enable delay extraction')
    parser.add_argument('--central_orchestration', action='store_true', help='Enable central orchestration')
    parser.add_argument('--determine_automatically', action='store_true', help='Enable automatic determination of simulation parameters')

    # Simulation parameters
    parser.add_argument('--num_simulations', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--num_cores', type=int, default=1, help='Number of CPU cores to use')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    start_time = time.time()

    warnings.filterwarnings("ignore")
    args = parse_arguments()
    
    # Process arguments
    train_and_test = not bool(args.log_path)
    column_names = {
        args.case_id: 'case_id',
        args.activity_name: 'activity_name',
        args.resource_name: 'resource',
        args.end_timestamp: 'end_timestamp',
        args.start_timestamp: 'start_timestamp'
    }
    
    # Set paths
    PATH_LOG = args.train_path if train_and_test else args.log_path
    PATH_LOG_test = args.test_path
    
    # Feature flags
    discover_extr_delays = discover_delays = args.extr_delays
    central_orchestration = args.central_orchestration
    determine_automatically = args.determine_automatically

    params = {
        'discover_extr_delays': discover_extr_delays,
        'discover_parallel_work': False,
        'central_orchestration': central_orchestration,
        'determine_automatically': determine_automatically,
        'PATH_LOG': PATH_LOG,
        'PATH_LOG_test': PATH_LOG_test,
        'train_and_test': train_and_test,
        'column_names': column_names,
        'num_simulations': args.num_simulations,
        'multiprocessing': True,
        'num_cpu': args.num_cores,
        'plot_on': False
    }

    simulator = AgentSimulator(params)
    simulator.execute_pipeline()

    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Execution time: {hours}h {minutes}m {seconds}s")