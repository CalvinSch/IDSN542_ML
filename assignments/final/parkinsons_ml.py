'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Final Project Part 1
'''

from pathlib import Path
import pandas as pd
import sys


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
	
    # Check correlation of features with diagnosis - Drop Dr ID because it's redacted anyways
	calculate_correlation_of_features(data=data.drop(columns=["DoctorInCharge"]), target_column='Diagnosis')

main()

