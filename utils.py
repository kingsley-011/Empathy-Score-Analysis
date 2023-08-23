import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt

# List of columns to be dropped from the dataset
drop_columns = ['Mouse position X', 'Mouse position Y', 'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
                'Event', 'Event value',
                'Computer timestamp', 'Export date', 'Recording date',
                'Recording date UTC', 'Recording start time', 'Timeline name', 'Recording Fixation filter name',
                'Recording software version', 'Recording resolution height', 'Recording resolution width',
                'Recording monitor latency', 'Presented Media width', 'Presented Media height',
                'Presented Media position X (DACSpx)', 'Presented Media position Y (DACSpx)', 'Original Media width',
                'Recording start time UTC', 'Original Media height', 'Sensor']
# Columns listed here will be dropped from the dataset using DataFrame.drop()

# List of columns containing string values (potentially non-numeric)
str_columns = ['Gaze direction left X', 'Gaze direction left Y', 'Gaze direction left Z',
                'Gaze direction right X', 'Gaze direction right Y', 'Gaze direction right Z',
                'Eye position left X (DACSmm)', 'Eye position left Y (DACSmm)', 'Eye position left Z (DACSmm)',
                'Eye position right X (DACSmm)', 'Eye position right Y (DACSmm)', 'Eye position right Z (DACSmm)',
                'Gaze point left X (DACSmm)', 'Gaze point left Y (DACSmm)', 'Gaze point right X (DACSmm)',
                'Gaze point right Y (DACSmm)', 'Gaze point X (MCSnorm)', 'Gaze point Y (MCSnorm)',
                'Gaze point left X (MCSnorm)', 'Gaze point left Y (MCSnorm)', 'Gaze point right X (MCSnorm)',
                'Gaze point right Y (MCSnorm)', 'Pupil diameter left', 'Pupil diameter right']
# Columns listed here are expected to contain string values (potentially non-numeric)
# These columns might need to be converted to appropriate data types for analysis


fill_columns = ['Pupil diameter left', 'Pupil diameter right', 'Fixation point X', 'Fixation point Y']

def process_data(df):
    df[fill_columns] = df[fill_columns].ffill()
    for i in range(len(str_columns)):
        df[str_columns[i]] = pd.to_numeric(df[str_columns[i]].str.replace(',', '.'), errors='coerce')
    return df

def add_data_specs(data, id):
    # Filter out rows with valid eye movement data
    valid_data = data[(data['Validity left'] == 'Valid') & (data['Validity right'] == 'Valid')]

    # Filter fixation data and calculate related statistics
    fixation_data = data[data['Eye movement type'] == 'Fixation']
    num_fixation = len(fixation_data)
    mean_fix_duration = fixation_data['Gaze event duration'].mean()

    # Calculate statistics for pupil diameter, gaze point, and fixation point columns
    pupil_diameter_stats = data[['Pupil diameter left', 'Pupil diameter right']].mean(axis=1).agg(['mean', 'median', 'std']).rename(lambda x: f'Pupil Diameter {x.capitalize()}')
    gaze_point_x_stats = data['Gaze point X'].agg(['mean', 'median', 'std']).rename(lambda x: f'Gaze Point X {x.capitalize()}')
    gaze_point_y_stats = data['Gaze point Y'].agg(['mean', 'median', 'std']).rename(lambda x: f'Gaze Point Y {x.capitalize()}')
    fixation_point_x_stats = data['Fixation point X'].agg(['mean', 'median', 'std']).rename(lambda x: f'Fixation Point X {x.capitalize()}')
    fixation_point_y_stats = data['Fixation point Y'].agg(['mean', 'median', 'std']).rename(lambda x: f'Fixation Point Y {x.capitalize()}')

    # Create a dictionary containing summary data
    summary_data = {
        'Participant Name': data['Participant name'].iloc[0],
        'Project Name': id,
        'Recording Name': data['Recording name'].iloc[0],
        'Total Fixations': num_fixation,
        'Avg. Fixation Duration': mean_fix_duration
    }
    # Update the dictionary with calculated statistics
    summary_data.update(pupil_diameter_stats)
    summary_data.update(gaze_point_x_stats)
    summary_data.update(gaze_point_y_stats)
    summary_data.update(fixation_point_x_stats)
    summary_data.update(fixation_point_y_stats)

    # Create a DataFrame using the summary data dictionary
    summary = pd.DataFrame(summary_data, index=[0])

    return summary


def empathy_score(data):
    # Calculate the mean of original and predicted empathy scores for each participant
    mean_val = data.groupby('Participant Name').agg({'Original Empathy Score': 'first', 'Predicted Empathy Score': 'mean'})

    # Reshape the DataFrame for visualization
    alt_df = mean_val.reset_index().melt(id_vars=['Participant Name'], value_vars=['Original Empathy Score', 'Predicted Empathy Score'], var_name='Score Type', value_name='Score')

    # Select a sample of participants for visualization
    sample_participants = alt_df['Participant Name'].unique()[:5]
    filter_alt_df = alt_df[alt_df['Participant Name'].isin(sample_participants)]

    # Create a bar plot using Seaborn
    plt.figure(figsize=(10, 5))
    sns.barplot(data=filter_alt_df, x='Participant Name', y='Score', hue='Score Type')

    # Add title and labels to the plot
    plt.title('Bar Plot of Actual and Predicted Empathy Scores for the First few Participants')
    plt.xlabel('Participant Name')
    plt.ylabel('Empathy Score')

    # Display the plot
    plt.show()


def train(data):
    drop_columns = ['Total Score extended', 'Project Name', 'Recording Name']
    x = data.drop(columns=drop_columns)
    y = data['Total Score extended']
    final_df = pd.DataFrame(columns=['Participant Name', 'Actual EmScore', 'Predicted EmScore'])
    le = LabelEncoder()
    x['Participant Name'] = le.fit_transform(x['Participant Name'])
    ids = data['Participant Name']
    n_samples = 30
    kfold = GroupKFold(n_samples)

    for i, (itrain, itest) in enumerate(kfold.split(x, y, groups=ids)):
        x_train, x_test, y_train, y_test = x.iloc[itrain], x.iloc[itest], y.iloc[itrain], y.iloc[itest]
        model = SVR(kernel='sigmoid', C=2.0, epsilon=0.7)
        model.fit(x_train, y_train)
        y_preds = model.predict(x_test)

        for j, (org, pred) in enumerate(zip(y_test, y_preds)):
            name = data.iloc[itest[j]]['Participant Name']
            final_df = final_df.append({'Participant Name': name, 'Actual EmScore': org, 'Predicted EmScore': pred}, ignore_index=True)
    return final_df


