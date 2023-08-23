import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from utils import process_data, add_data_specs, empathy_score, train

# Use the glob library to search for CSV files in the './EyeT/' directory
data_files = glob.glob('./EyeT/*.csv')

# Initialize an empty list to store summary DataFrames
df_list = []

# Iterate through the list of data files
for f in data_files:
    data = pd.read_csv(f, usecols=lambda column: column != 0, low_memory=True)
    processed_data = process_data(data)
    file_name = os.path.basename(f)
    if file_name.startswith('EyeT_group_dataset_III_'):
        group = 'Test group experiment'
    elif file_name.startswith('EyeT_group_dataset_II_'):
        group = 'Control group experiment'
    summary = add_data_specs(processed_data, group)
    df_list.append(summary)

print(df_list)
# Concatenate all the summary DataFrames in the df_list into a single DataFrame
df = pd.concat(df_list, ignore_index=True)

# Load and process questionnaire data
new_df = pd.read_csv('Questionnaire_dataset.csv', encoding='cp1252')
new_df[['Total Score extended']] = new_df[['Total Score extended']].astype(object)

# Merge the questionnaire data with the eye-tracking data
columns_to_append = new_df[['Total Score extended']]
result = pd.merge(df, columns_to_append, left_index=True, right_index=True, how='left')

# Save the merged DataFrame to a CSV file
result.to_csv('output_data.csv', index=False)

# Visualize empathy scores
empathy_score(result)

# Group data by project names
unique_projects = result['Project Name'].unique()
project_dfs = {}

for project in unique_projects:
    project_data = result[result['Project Name'] == project]
    project_dfs[project] = project_data

# Train models and visualize results
control_group = project_dfs['Control group experiment']
test_group = project_dfs['Test group experiment']

control_group_results = train(control_group)

test_group_results = train(test_group)

test_group_pupil_results = train(test_group)
