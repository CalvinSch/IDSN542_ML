'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Final Project Part 1 + 2 + 3
'''

from pathlib import Path
import pandas as pd
import sys

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit


def load_data(path):
	# Load Parkinson's Disease csv from path or default location
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
	# plt.show()
	pass

def training_accuracy(model, X_train, y_train):
	predictions = model.predict(X_train)
	accuracy = (predictions == y_train).mean()
	return accuracy

def plot_roc(model, X, y, model_name):
	y_scores = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class
	fpr, tpr, thresholds = roc_curve(y, y_scores)
	plt.plot(fpr, tpr, linewidth=2, label=model_name)
	plt.plot([0, 1], [0, 1], 'k--')  # Plots 45 degree dashed line as black
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend()
	# plt.show()

def caclulate_training_rmse(training_predictions, target):
	return root_mean_squared_error(target, training_predictions)

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
	categorical_attributes = [col for col in data.columns if set(data[col]).issubset({0, 1, 2, 3})] # Categorical data encoded as 0,1,2,3 (yes, no, or level of education/ethnicity)
	numerical_attributes = [col for col in data.columns if col not in categorical_attributes] # Rest are numerical attributes
	print("\nCategorical attributes:", categorical_attributes)
	print("\nNumerical attributes:", numerical_attributes)
	
	# Plotting numerical attributes for correlation comparison - control how correlated value should be in order to make it on to plot
	plot_correlation_comparisons(data, numerical_attributes, corr_data, threshold=0.01)
	
	# Let's only use features with correlation above 0.01 for modeling
	selected_attributes = [attr for attr in numerical_attributes + categorical_attributes if abs(corr_data[attr]) > 0.01]
	X = data[selected_attributes].copy()
	y = data['Diagnosis'].copy()
	
	# Split data into training and test sets (pd stands for parkinsons disease)
	split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
	for train_index, test_index in split.split(X, y):
		parkinsons_train = X.iloc[train_index]
		parkinsons_test = X.iloc[test_index]
		diagnosis_train = y.iloc[train_index]
		diagnosis_test = y.iloc[test_index]

	# Pipelines for numerical and categorical data
	num_pipeline = Pipeline([
		('std_scaler', StandardScaler()), # Scale numerical data
	])

	cat_pipeline = Pipeline([
		('one_hot', OneHotEncoder()), # Cateogrical data is ranked values so one-hot encode
	])

	# Run the full pipeline on our selected attributes (not the full list)
	full_pipeline = ColumnTransformer([
	    ("num", num_pipeline, [attr for attr in selected_attributes if attr in numerical_attributes]),
	    ("cat", cat_pipeline, [attr for attr in selected_attributes if attr in categorical_attributes]),
	])

	parkinsons_train_prepared = full_pipeline.fit_transform(parkinsons_train)

	### MODEL 1: DECISION TREE CLASSIFIER
	decision_tree_clf = DecisionTreeClassifier(random_state=42)
	dt_param_grid = {
		'criterion': ['gini', 'entropy', 'log_loss'],
		'splitter': ['best', 'random'],
		'max_depth': [None, 5, 10, 20, 30],
		'min_samples_split': [2, 5, 10],
		'min_samples_leaf': [1, 2, 4],
		'max_features': [None, 'sqrt', 'log2'],
		'class_weight': [None, 'balanced'],
	}
	
	# grid search
	dt_grid_search = GridSearchCV( decision_tree_clf, dt_param_grid, cv=5, scoring='neg_mean_squared_error',return_train_score=True)
	dt_grid_search.fit( parkinsons_train_prepared, diagnosis_train)
	print("Best parameters for Decision Tree Classifier:", dt_grid_search.best_params_)
	dt_best = dt_grid_search.best_estimator_

	print(dt_best.predict_proba(parkinsons_train_prepared)[:10])
	print("Diagnosis mean:", diagnosis_train.mean())
	print("Predictions mean:", dt_best.predict(parkinsons_train_prepared).mean())
	
	plot_roc(dt_best, parkinsons_train_prepared, diagnosis_train, "Decision Tree Classifier")

	# Calculate RMSE after hyperparam search
	print("Decision Tree Training RMSE: " + str(caclulate_training_rmse(dt_best.predict_proba(parkinsons_train_prepared)[:, 1], diagnosis_train)))
	
	### MODEL 2: RANDOM FOREST CLASSIFIER
	random_forest_clf = RandomForestClassifier(random_state=42)
	rf_param_grid = [
	{'n_estimators': [5,15,30], 'max_features': [2,4,8,15]},
	{'bootstrap': [False], 'n_estimators': [10, 40], 'max_features': [3,8,20]},] # Using param grid from class examples
	
	rf_grid_search = GridSearchCV( random_forest_clf, rf_param_grid, cv=5, scoring='neg_mean_squared_error',return_train_score=True)
	rf_grid_search.fit( parkinsons_train_prepared, diagnosis_train)
	print("Best parameters for Random Forest Classifier:", rf_grid_search.best_params_)
	rf_best = rf_grid_search.best_estimator_
	plot_roc(rf_best, parkinsons_train_prepared, diagnosis_train, "Random Forest Classifier")

	# Calculate RMSE after hyperparam search
	print("Decision Tree Training RMSE: " + str(caclulate_training_rmse(rf_best.predict_proba(parkinsons_train_prepared)[:, 1], diagnosis_train)))

	### MODEL 3: SUPPORT VECTOR CLASSIFIER
	svc_clf = SVC(random_state=42, probability=True)
	svc_param_grid = {
		'C': [0.1, 1, 10],
		'kernel': ['linear', 'rbf', 'poly'],
		'gamma': ['scale', 'auto'],
		'class_weight': [None, 'balanced'],
		'probability': [True],
	}
	svc_grid_search = GridSearchCV( svc_clf, svc_param_grid, cv=5, scoring='neg_mean_squared_error',return_train_score=True)
	svc_grid_search.fit( parkinsons_train_prepared, diagnosis_train)
	print("Best parameters for Support Vector Classifier:", svc_grid_search.best_params_)
	svc_best = svc_grid_search.best_estimator_
	plot_roc(svc_best, parkinsons_train_prepared, diagnosis_train, "Support Vector Classifier")

	# Calculate RMSE after hyperparam search
	print("Decision Tree Training RMSE: " + str(caclulate_training_rmse(svc_best.predict_proba(parkinsons_train_prepared)[:, 1], diagnosis_train)))

main()
