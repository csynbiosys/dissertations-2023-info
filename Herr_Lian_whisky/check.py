import pandas as pd
import numpy as np

# Load the merged DataFrame from the Excel file
df = pd.read_excel('merged_data.xlsx')


# Define a function to check if two columns are equal or lack data
def check_columns(row, col1, col2):
    if row[col1] == 'not show' or pd.isnull(row[col2]):
        return 'lack data'
    elif row[col1] == row[col2]:
        return True
    else:
        return False


# Define a function to check the distillery status
def check_status(row):
    if row['Distillery Staus'] == 'not show' or pd.isnull(row['status_data']):
        return 'lack data'
    elif (row['Distillery Staus'] == 'Operational' and row['status_data'] == 'Active') or \
            (row['Distillery Staus'] == 'Closed' and row['status_data'] == 'Closed/Dismantled'):
        return True
    else:
        return False


# Create the 'region_check' and 'status_check' columns
df['region_check'] = df.apply(check_columns, args=('region', 'region_data'), axis=1)
df['status_check'] = df.apply(check_status, axis=1)

# Save the DataFrame to a new Excel file
df.to_excel('checked_data.xlsx', index=False)
