import pandas as pd

# Load the DataFrame from the Excel file
df = pd.read_excel('checked_data.xlsx')

# Calculate the counts of each value in the 'region_check' and 'status_check' columns
region_counts = df['region_check'].value_counts(normalize=True)
status_counts = df['status_check'].value_counts(normalize=True)

# Print the probabilities
print("Probabilities for 'region_check':")
print(region_counts)
print("\nProbabilities for 'status_check':")
print(status_counts)