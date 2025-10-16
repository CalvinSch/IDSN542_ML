'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Homework 4
'''

# NOTE All plots and data prints from the previous lab are commented out or removed for clarity
# I'm only printing the model data for this homework.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

from zlib import crc32

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import OneHotEncoder

'''
Regression dataset of GPU kernel parameters and their runtimes. Courtesy of:

 -- Rafael Ballester-Ripoll, Enrique G. Paredes, Renato Pajarola.
    Sobol Tensor Trains for Global Sensitivity Analysis.
    In arXiv Computer Science / Numerical Analysis e-prints, 2017
    (https://128.84.21.199/abs/1712.00233).

    -- Cedric Nugteren and Valeriu Codreanu. CLTune: A Generic Auto-Tuner for OpenCL Kernels.
    In: MCSoC: 9th International Symposium on Embedded Multicore/Many-core Systems-on-Chip. IEEE, 2015
    (http://ieeexplore.ieee.org/document/7328205/)

 Attribute Information
  -- Independent variables:
    1-2. MWG, NWG: per-matrix 2D tiling at workgroup level: {16, 32, 64, 128} (integer)
    3. KWG: inner dimension of 2D tiling at workgroup level: {16, 32} (integer)
    4-5. MDIMC, NDIMC: local workgroup size: {8, 16, 32} (integer)
    6-7. MDIMA, NDIMB: local memory shape: {8, 16, 32} (integer)
    8. KWI: kernel loop unrolling factor: {2, 8} (integer)
    9-10. VWM, VWN: per-matrix vector widths for loading and storing: {1, 2, 4, 8} (integer)
    11-12. STRM, STRN: enable stride for accessing off-chip memory within a 
           single thread: {0, 1} (categorical)
    13-14. SA, SB: per-matrix manual caching of the 2D workgroup tile: {0, 1} (categorical)
  -- Total of 241600 possible parameter configurations
  -- Output:
    15-18. Run1, Run2, Run3, Run4: performance times in milliseconds for 4 independent
           runs using the same parameters. They range between 13.25 and 3397.08.   
'''

def load_gpu_data():
    return pd.read_csv("sgemm_product.csv")

def split_train_test(data, test_ratio): #picks indices at random each time you run it
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# Data authors suggest using log of runtimes and using average runtime as target variable
def add_log_runtimes_and_average(data):
    data["avg_runtime"] = data[["Run1 (ms)","Run2 (ms)","Run3 (ms)","Run4 (ms)"]].mean(axis=1)
    data["log_avg_runtime"] = np.log(data["avg_runtime"])
    # Also add a categorical variable for stratified sampling
    data["runtime_cat"] = pd.cut(
        data["avg_runtime"],
        bins=[0, 50, 100, 200, 400, 800, np.inf],
        labels=[1, 2, 3, 4, 5, 6]
    )
    plt.figure()
    data["runtime_cat"].hist()
    plt.title('Distribution of Runtime Categories')
    plt.xlabel('Runtime Category')
    plt.ylabel('Number of Samples')
    #plt.show()

def stratified_split(data):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["runtime_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set

def remove_runtime_cat(strat_train_set, strat_test_set):
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("runtime_cat", axis=1, inplace=True)


def main():
    gpu_data = load_gpu_data() # Load data (no missing data)

    # Confirm no missing data by checking for NaNs
    # print(gpu_data.isnull().sum())

    add_log_runtimes_and_average(gpu_data) # Add relevant attributes
    gpu_data = gpu_data.reset_index() # Add index column for reliable train/test split

    train_set, test_set = split_train_test(gpu_data, 0.2) # Randomized training and test sests
    strat_train_set, strat_test_set = stratified_split(gpu_data) # Stratified training and test sets

    # print("Random sampling:")
    # print(len(train_set))
    # print(train_set.head(5))
    # print(len(test_set))
    # print(test_set.head(5))

    # print("\nStratified sampling:")
    # print(len(strat_train_set))
    # print(strat_train_set.head(5))
    # print(len(strat_test_set))
    # print(strat_test_set.head(5))

    # Remove the runtime_cat attribute so we can use original data for ML
    remove_runtime_cat(strat_train_set, strat_test_set)
    gpu_data = strat_train_set.copy()

    # Now that we've split, encode categorical variables using OneHotEncoder
    categorical_features = ["STRM", "STRN", "SA", "SB"]
    one_hot_encoder = OneHotEncoder()
    encoded_categorical = one_hot_encoder.fit_transform(gpu_data[categorical_features])

    # See correlation of numeric attributes with avg_runtime
    numeric_cols = [col for col in gpu_data.columns if col not in categorical_features]
    corr_matrix = gpu_data[numeric_cols].corr()
    # print(corr_matrix["avg_runtime"].sort_values(ascending=False))

    # See correlation of numeric attributes with log_avg_runtime
    numeric_cols = [col for col in gpu_data.columns if col not in categorical_features]
    corr_matrix = gpu_data[numeric_cols].corr()
    # print(corr_matrix["log_avg_runtime"].sort_values(ascending=False))

    # Visualize correlations with a scatter matrix, focusing on the most correlated attributes (> 0.1)
    correlated_attributes = ["MWG", "NWG", "VWM", "NDIMC", "MDIMC", "VWN", "avg_runtime"]
    scatter_matrix(gpu_data[correlated_attributes], figsize=(12,8))
    #plt.show() # You have to close the plot window to continue the program

    # Create hexbin plots for better density visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, attr in enumerate(["MWG", "NWG", "VWM", "VWN", "NDIMC", "MDIMC"]):
        hb = axes[idx].hexbin(gpu_data[attr], 
                            gpu_data["avg_runtime"],
                            gridsize=20,
                            cmap='YlOrRd')
        axes[idx].set_xlabel(attr)
        axes[idx].set_ylabel('avg_runtime')
        fig.colorbar(hb, ax=axes[idx])
    
    plt.tight_layout()
    #plt.show()

    # Try a log transform of the correlated attributes since they're all powers of 2
    log_correlated_data = gpu_data[correlated_attributes].apply(np.log2)
    scatter_matrix(log_correlated_data, figsize=(12,8))
    #plt.show()

    corr_matrix_log = log_correlated_data.corr()
    # print("\nCorrelation matrix for log2 transformed data:")
    # print(corr_matrix_log)

    ''' BEGIN HW04 '''
    print("\n--- HW04 ---\n")

    # Create pipeline to do data transforms of numeric and categorical data on training data
    numeric_cols = [col for col in gpu_data.columns if col not in categorical_features and col not in ["log_avg_runtime", "avg_runtime", "Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)", "index"]]
    full_pipeline = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_cols),
        ("categorical", OneHotEncoder(), categorical_features)
    ]) # No missing values, so only using StandardScaler for numeric data. 

    # Copy "answer" attribute and remove from prepared training data
    gpu_data_label = gpu_data["avg_runtime"].copy()
    gpu_data_log_label = gpu_data["log_avg_runtime"].copy()
    gpu_data = gpu_data.drop(["log_avg_runtime", "avg_runtime", "Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"], axis=1)
    
    gpu_data_prepared = full_pipeline.fit_transform(gpu_data)
    print(gpu_data_prepared)
    print(gpu_data)

    # Cross Validation Training of Linear Regression
    linear_regression = LinearRegression()
    linear_regression.fit(gpu_data_prepared, gpu_data_label)

    gpu_linear_predictions = linear_regression.predict(gpu_data_prepared)
    linear_mse = mean_squared_error(gpu_data_label, gpu_linear_predictions)
    linear_rmse = np.sqrt(linear_mse)
    linear_scores = cross_val_score(linear_regression, gpu_data_prepared, gpu_data_label,
                                    scoring='neg_mean_squared_error', cv=10)
    linear_rmse_scores = np.sqrt(-linear_scores)
    print("Training Linear Regression RMSE:", linear_rmse)
    print("Training Linear Regression Cross-Validation Scores:", linear_rmse_scores)
    
    # Cross Validation Training of Random Forest Regressor
    random_forest = RandomForestRegressor()
    random_forest.fit(gpu_data_prepared, gpu_data_label)

    gpu_forest_predictions = random_forest.predict(gpu_data_prepared)
    forest_mse = mean_squared_error(gpu_data_label, gpu_forest_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_scores = cross_val_score(random_forest, gpu_data_prepared, gpu_data_label,
                                    scoring='neg_mean_squared_error', cv=5)
    forest_rmse_scores = np.sqrt(-forest_scores)
    print("Training Random Forest Regression RMSE:", forest_rmse)
    print("Training Random Forest Regression Cross-Validation Scores:", forest_rmse_scores)

    # Now that we've trained and cross validated, we can evaluate on the test set
    gpu_test_set = strat_test_set.copy() # Copy training data
    gpu_test_labels = gpu_test_set["avg_runtime"].copy() # Get our "answer" attribute
    gpu_test_set = gpu_test_set.drop(["log_avg_runtime", "avg_runtime", "Run1 (ms)", "Run2 (ms)", "Run3 (ms)", "Run4 (ms)"], axis=1) # Remove "answers" from test
    gpu_test_prepared = full_pipeline.transform(gpu_test_set)

    final_linear_predictions = linear_regression.predict(gpu_test_prepared)
    final_forest_predictions = random_forest.predict(gpu_test_prepared)
   
    final_linear_mse = mean_squared_error(gpu_test_labels, final_linear_predictions)
    final_linear_rmse = np.sqrt(final_linear_mse)
    final_forest_mse = mean_squared_error(gpu_test_labels, final_forest_predictions)
    final_forest_rmse = np.sqrt(final_forest_mse)
    
    print("\nFinal Linear Regression Test RMSE:", final_linear_rmse)
    print("Final Random Forest Regression Test RMSE:", final_forest_rmse)

    ''' Further Analysis: Log-Transformed Target Variable '''
    print("\nTraining and Cross-Validation with Log of Average Runtime")
    linear_regression_log = LinearRegression()
    
    # Training phase
    linear_regression_log.fit(gpu_data_prepared, gpu_data_log_label)
    training_log_predictions = linear_regression_log.predict(gpu_data_prepared)
    training_log_mse = mean_squared_error(gpu_data_log_label, training_log_predictions)
    training_log_rmse = np.sqrt(training_log_mse)
    print("Training Linear Regression RMSE (log):", training_log_rmse)
    
    # Cross-validation phase
    cv_log_scores = cross_val_score(linear_regression_log, 
                                  gpu_data_prepared, 
                                  gpu_data_log_label,
                                  scoring='neg_mean_squared_error', 
                                  cv=10)
    cv_log_rmse_scores = np.sqrt(-cv_log_scores)
    print("Training Linear Regression Cross-validation Scores (log):", cv_log_rmse_scores.mean(), "±", cv_log_rmse_scores.std())

    # Test phase
    print("\nFinal Test Linear Regression Evaluation (log)")
    gpu_test_log_labels = strat_test_set["log_avg_runtime"].copy()
    test_log_predictions = linear_regression_log.predict(gpu_test_prepared)
    test_log_mse = mean_squared_error(gpu_test_log_labels, test_log_predictions)
    test_log_rmse = np.sqrt(test_log_mse)
    print("FInal Test Linear Regression RMSE (log space):", test_log_rmse)

main()