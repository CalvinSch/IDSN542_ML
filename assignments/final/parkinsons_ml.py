'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Final Project Part 1 + 2
'''

from pathlib import Path
import pandas as pd
import sys

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def load_data(path):
    # Load Parkinson's Disease csv from path or default location
	if path is None:
		path = Path(__file__).resolve().parent / "parkinsons_disease_data.csv"
	else:
		path = Path(path)

	if not path.exists():
		raise FileNotFoundError(f"File not found: {path}")

	df = pd.read_csv(path)
	return df


def check_missing_data(data):
    # Check for missing data and print per attribute
	total_rows = len(data)
	missing_count = data.isnull().sum()

	print(f"Data shape: {data.shape[0]} rows, {data.shape[1]} columns")
	print("\nMissing values per column (counts):")

	missing_dict = {}
	for col, cnt in missing_count.items():
		missing_dict[col] = int(cnt)
		print(f"{col}: {int(cnt)}")

	cols_with_missing = {c: v for c, v in missing_dict.items() if v > 0}
	if not cols_with_missing:
		print('\nNo missing values found.')
	else:
		print('\nColumns with missing values:')
		for c, v in cols_with_missing.items():
			print(f"{c}: {v}")

	return missing_dict

def calculate_correlation_of_features(data, target_column):
	# Calculate the correlation of the given attributes with the diagnosis target
	if target_column not in data.columns:
		print(f"Target column '{target_column}' not found in data.")
		return
	corr_matrix = data.corr()
	target_corr = corr_matrix[target_column].sort_values(ascending=False)
	print(f"\nCorrelation of features with '{target_column}':")
	print(target_corr)
	return target_corr
	
def plot_correlation_comparisons(data, attributes, correlation_values, threshold):
    # Visualize correlations with a scatter matrix, focusing on the most correlated attributes (> 0.1)
    correlated_attributes = [attr for attr in attributes if abs(correlation_values[attr]) > threshold]
    print("\nAttributes with correlation above threshold", threshold, ":", correlated_attributes)
    n_attrs = len(correlated_attributes)
    # Set figure size: width and height proportional to number of attributes
    figsize = (2.5 * n_attrs, 2.5 * n_attrs)
    scatter_matrix(data[correlated_attributes], figsize=figsize)
    plt.show() # You have to close the plot window to continue the program
    pass


def main():
	# Load data
	try:
		data = load_data("parkinsons_disease_data.csv")
	except FileNotFoundError as e:
		print(e)
		sys.exit(1)

	print('\nFirst 3 rows:')
	print(data.head(3))

	# Check for missing data
	check_missing_data(data)
	
    # Drop irrelevant columns
	data = data.drop(columns=["PatientID", "DoctorInCharge"])
	
	# Check correlation of features with diagnosis - Drop Dr ID because it's redacted anyways
	corr_data = calculate_correlation_of_features(data, target_column='Diagnosis')

	# Create lists of numerical attributes and categorical attributes
	categorical_attributes = [col for col in data.columns if set(data[col]).issubset({0, 1, 2, 3})] # Categorical data encoded as 0,1,2,3 (yes, no, or level of education)
	numerical_attributes = [col for col in data.columns if col not in categorical_attributes] # Rest are numerical attributes
	print("\nCategorical attributes:", categorical_attributes)
	print("\nNumerical attributes:", numerical_attributes)
	
	# Plotting numerical attributes for correlation comparison - control how correlated value should be in order to make it on to plot
	plot_correlation_comparisons(data, numerical_attributes, corr_data, threshold=0.1)
main()

