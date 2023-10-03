import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import trimboth
from sklearn.cluster import KMeans


class DataPreparation:
    """
    Defines each data quality issue as a function.
    Return type of functions are the actual issue count/issues for unittest purposes.
    """
    def __init__(self, df, y_label=None) -> None:
        self.logger = logging.getLogger(__name__)
        self.df = df
        self.y_label = y_label

    def get_numerical_vars(self):
        numerical_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if self.y_label in numerical_cols:
            numerical_cols.remove(self.y_label)
        return self.df[numerical_cols]

    def report_differently_distributed_feature(self, max_deviance = 2, trim_proportion = 0.1):

        numerical_vars = self.get_numerical_vars()

        # Using vectorized operations for efficiency
        means = numerical_vars.mean()
        std_devs = numerical_vars.std()

        # Compute trimmed statistics
        def _get_trimmed_stats(values):
            sorted_vals = np.sort(values)
            trimmed_vals = trimboth(sorted_vals, trim_proportion)
            return np.mean(trimmed_vals), np.std(trimmed_vals)

        means_trimmed_mean, means_trimmed_std = _get_trimmed_stats(means)
        std_devs_trimmed_mean, std_devs_trimmed_std = _get_trimmed_stats(std_devs)

        # Use numpy to compute z-scores for vectorized operation
        mean_deviances = np.abs((means - means_trimmed_mean) / means_trimmed_std)
        std_dev_deviances = np.abs((std_devs - std_devs_trimmed_mean) / std_devs_trimmed_std)

        # Get columns where deviance exceeds the max_deviance for either mean or std deviation
        issues = numerical_vars.columns[(mean_deviances > max_deviance) | (std_dev_deviances > max_deviance)]

        for column in issues:
            self.logger.info(f"{column} differently distributed due to its statistical properties")

        return list(issues)

    def report_schema_violations(self, schema_path):
        if schema_path:
            # Checking if the file exists and is valid JSON
            if not os.path.exists(schema_path):
                raise ValueError("Schema path does not exist.")
            try:
                with open(schema_path, "r") as f:
                    schema = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON file.")

            violations = {}
            for column, _dtype in schema.items():
                if column not in self.df.columns:
                    violations[column] = "Column missing in dataset"
                elif self.df[column].dtype != _dtype:
                    violations[column] = f"Expected {_dtype} but found {self.df[column].dtype}"

            self.logger.info(f"Found {len(violations)} Schema Violations!")
            for column, issue in violations.items():
                logging.info("Schema Violation in %s: %s", column, issue)
            return violations


    def remove_constant_features(self):
        constant_columns = []

        # Iterate over each column
        for column in self.df.columns:
            # NOTE WE DONT DROP THE Y LABEL EVEN THO ITS CONSTANT!
            if column is not self.y_label: #
                # If unique value count is 1, then it's a constant column
                if self.df[column].nunique() == 1:
                    constant_columns.append(column)
                    self.logger.info(f"'{column}' detected as a constant feature.")
        self.df.drop(columns=constant_columns)

        return constant_columns

    def class_overlap_removal(self):
        if self.y_label is None:
            self.logger.info("Class overlap cannot be run since y-info is not there!")
            return 0
        else:
            self.logger.info("Starting class overlap removal process.")
            numerical_vars = self.get_numerical_vars()
            x = numerical_vars.values
            y = self.df[self.y_label].values

            positive_negative_ratio = np.sum(y == 1) / np.sum(y == 0)
            k = int(len(self.df) / np.sum(y == 1))
            self.logger.info(
                f"\tCalculated k value: {k}, positive-negative ratio: {positive_negative_ratio}"
            )

            random_state = 200
            kmeans_model = KMeans(n_clusters=k, random_state=random_state)
            cluster_predictions = kmeans_model.fit_predict(x)

            # Converting arrays to DataFrame
            merged_df = pd.DataFrame(
                data=np.hstack((x, y.reshape(-1, 1), cluster_predictions.reshape(-1, 1))),
                columns=list(numerical_vars.columns) + ["Y_LABEL", "CLUSTER_NUMBER"],
            )

            # Collecting indexes of rows to be kept
            indexes_to_keep = []
            for cluster_num in range(k):
                cluster_data = merged_df[merged_df["CLUSTER_NUMBER"] == cluster_num]
                n0 = cluster_data[cluster_data["Y_LABEL"] == 0].shape[0]
                n1 = cluster_data[cluster_data["Y_LABEL"] == 1].shape[0]

                if n0 == 0 or n1 / n0 >= positive_negative_ratio:
                    indexes_to_keep.extend(
                        cluster_data[cluster_data["Y_LABEL"] == 1].index.tolist()
                    )
                else:
                    indexes_to_keep.extend(
                        cluster_data[cluster_data["Y_LABEL"] == 0].index.tolist()
                    )
            overlapped_row_count = len(self.df) - len(indexes_to_keep)
            self.logger.info(f"\tFound {overlapped_row_count} overlapping rows")
            self.df = self.df.iloc[indexes_to_keep, :]
        return overlapped_row_count

    def remove_correlated_metrics(self, threshold=0.7):
        """
        Remove columns that are correlated with other columns.
        """
        # Compute the correlation matrix
        corr_matrix = self.get_numerical_vars().corr().abs()

        # Create a mask for the upper triangle of the matrix
        upper_triangle_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlated_cols = [col for col in corr_matrix.columns if any(corr_matrix.where(upper_triangle_mask)[col] > threshold)]

        self.logger.info(f"Found {len(correlated_cols)} correlated columns to be removed.")

        # Drop the correlated columns
        self.df.drop(columns=correlated_cols)

        return correlated_cols

    def remove_redundant_metrics(self, margin=0.001):
        """
        Remove columns that have almost zero standard deviation or are identical to others.
        """
        # Metrics with almost zero standard deviation
        numerical_vars = self.get_numerical_vars()
        redundant_std = numerical_vars.columns[numerical_vars.std() < margin]

        # Check for identical columns
        redundant_identical = []
        columns = numerical_vars.columns.tolist()
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:]):
                if numerical_vars[col1].equals(numerical_vars[col2]):
                    redundant_identical.append(col2)

        redundant_cols = list(set(redundant_std).union(set(redundant_identical)))
        self.logger.info(f"Found {len(redundant_cols)} redundant columns to be removed.")

        # Drop the redundant columns
        self.df.drop(columns=redundant_cols)
        return redundant_cols

    def fill_missing_values(self):
        # Impute missing values only for numerical columns
        num_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        total_missing = 0
        for col in num_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                self.logger.info(f"Imputed {missing_count} missing values in column '{col}' with median.")
                total_missing += missing_count
        # For non-numerical columns, use back-fill method
        non_num_cols = self.df.select_dtypes(exclude=["float64", "int64"]).columns
        for col in non_num_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                self.df.fillna(method="bfill", inplace=True)
                self.logger.info(f"Imputed {missing_count} missing values in column '{col}' using back-fill method")
                total_missing += missing_count
        return total_missing

    def remove_duplicates(self):
        duplicated_count = self.df.duplicated().sum()
        self.logger.info(f"Found {duplicated_count} duplicated rows.")
        self.df = self.df.drop_duplicates()

    def remove_outliers(self, outlier_threshold=20):
        """
        :param outlier_threshold: should be greater than 1.
                if its 1 then everything not in the interquartile range will be removed
        :return:
        """
        non_outliers = ~self.df.index.isin([])  # This will select all rows initially
        for column in self.df.select_dtypes(include=["float64", "int64"]):
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers_mask = (self.df[column] < (Q1 - outlier_threshold * IQR)) | (
                self.df[column] > (Q3 + outlier_threshold * IQR)
            )
            non_outliers &= ~outliers_mask
            if outliers_mask.sum() > 0:
                self.logger.info(f"Found {outliers_mask.sum()} outliers in {column}.")

        self.df = self.df[non_outliers]


    #  -------------------------LEGACY CODE----------------------------
    # def assess_data_quality(self):
    #     """Assesses data quality by logging information about the dataset."""
    #     self.logger.info(f"Accessing data quality of a dataset of shape {self.df.shape}")
    #
    #     # Logging missing values
    #     missing_values = self.df.isnull().sum()
    #     self.logger.info(f"Missing values:\n{missing_values}")
    #
    #     # Logging duplicates
    #     duplicate_count = self.df[self.df.duplicated()].shape[0]
    #     self.logger.info(f"Found {duplicate_count} duplicate rows.")
    #
    #     # Logging outliers
    #     for column in self.df.select_dtypes(include=['float64', 'int64']):
    #         Q1 = self.df[column].quantile(0.25)
    #         Q3 = self.df[column].quantile(0.75)
    #         IQR = Q3 - Q1
    #         outlier_count = self.df[(self.df[column] < (Q1 - 1.5 * IQR)) | (self.df[column] > (Q3 + 1.5 * IQR))].shape[0]
    #         self.logger.info(f"Found {outlier_count} outliers in {column}.")
    #
    # def auto_clean(self):
    #     """
    #     Automatically cleans the dataset in a pre-defined order.
    #     Note: this order chan be changed as per the data cleaning use-case
    #     """
    #     self.remove_duplicates()
    #     self.fill_missing_values()
    #     self.remove_outliers()
    #     self.remove_correlated_metrics()
    #     self.remove_redundant_metrics()
    #     self.class_overlap_removal()