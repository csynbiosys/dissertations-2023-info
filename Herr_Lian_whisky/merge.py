import pandas as pd

# Load the two DataFrames from the Excel files
df1 = pd.read_excel('auction_data.xlsx')
df2 = pd.read_excel('distillery_data.xlsx')

# Rename the columns in the second DataFrame
df2 = df2.rename(columns={'Status': 'status_data', 'Region': 'region_data'})

# Merge the two DataFrames on the 'distillery' column
merged_df = pd.merge(df1, df2, on='distillery', how='left')

# Save the merged DataFrame to a new Excel file
merged_df.to_excel('merged_data.xlsx', index=False)
