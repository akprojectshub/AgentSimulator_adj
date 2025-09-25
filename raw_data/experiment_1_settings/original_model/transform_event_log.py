import pandas as pd

# Load the CSV file
df = pd.read_csv('original_model/experiment_1_bimp_log_original.csv')  # Replace with your actual file path

# Remove rows where Activity is 'Start' or 'End'
df = df[~df['Activity'].isin(['Start', 'End'])]

# Rename columns
df = df.rename(columns={
    'Case ID': 'case_id',
    'Activity': 'activity',
    'Resource': 'resource',
    'Start Timestamp': 'start_time',
    'Complete Timestamp': 'end_time'
})

# Optionally, save the cleaned DataFrame to a new CSV
df.to_csv('experiment_1_bimp_log.csv', index=False)

print("Filtered and renamed data saved to 'experiment_1_bimp_log.csv'")