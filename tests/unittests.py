import unittest

import numpy as np
import pandas as pd

from DataPreparation import DataPreparation
import cli_tool


class TestSchemaViolations(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.df = pd.read_csv("test.csv")
        self.orig_len = len(self.df)

    def test_detect_schema_violations(self):
        data = pd.DataFrame(
            {
                "Name": ["John", "Alice", "Bob", "Eve"],
                "Age": [25, 30, -5, np.nan],
                "Email": [
                    "john@example.com",
                    "alice@sample.net",
                    "bob@invalid.email",
                    "eve@example.com",
                ],
            }
        )
        dp_obj = DataPreparation(df=data)
        violations = dp_obj.report_schema_violations(schema_path="sample_schema.json")
        print(violations)
        self.assertIn("Email", violations)


    def test_duplicate(self):
        dp_obj = DataPreparation(df=self.df, y_label='RealBug')
        dp_obj.remove_duplicates()
        self.assertLess(len(dp_obj.df), self.orig_len)


    def test_outliers(self):
        dp_obj = DataPreparation(df=self.df)
        dp_obj.remove_outliers()
        self.assertLess(len(dp_obj.df), self.orig_len)

    def test_missing_values(self):
        # TODO THIS TEST is broken
        # there are still missing values
        dp_obj = DataPreparation(df=self.df)
        total_missing = dp_obj.fill_missing_values()

        dp_obj.df.to_csv('missing_removed.csv')
        self.assertGreater(total_missing,0)


    def test_correlation(self):
        dp_obj = DataPreparation(df=self.df)
        correlated_cols = dp_obj.remove_correlated_metrics()
        # print(f"{len(self.df.columns)=}, {correlated_cols=}")
        self.assertGreater(len(correlated_cols),0)

    def test_redundancy(self):
        dp_obj = DataPreparation(df=self.df)
        redun_cols = dp_obj.remove_redundant_metrics()
        print(f"{len(self.df.columns)=}, {redun_cols=}")
        self.assertGreater(len(redun_cols),0)

    def test_class_overlap(self):
        dp_obj = DataPreparation(df=self.df, y_label='RealBug')
        # for class overlap to work, first we need to remove Nas in the dataset
        _ = dp_obj.fill_missing_values()
        overlapped_row_count = dp_obj.class_overlap_removal()
        self.assertGreater(overlapped_row_count, 10)

    def test_differently_distributed_feature(self):
        dp_obj = DataPreparation(df=self.df, y_label='RealBug')
        different_features = dp_obj.report_differently_distributed_feature()
        # print(f"{different_features=}")
        self.assertGreater(len(different_features), 1)

    def test_constant_feature(self):
        dp_obj = DataPreparation(df=self.df, y_label='RealBug')
        # I manually added a constant feature in the testing csv
        constant_features = dp_obj.remove_constant_features()
        self.assertEqual(len(constant_features),1)


    def test_clean_and_generate_report(self):
        cleaned_df = cli_tool.clean_and_generate_report(self.df, schema_path=None, y_label='RealBug')
        print(cleaned_df.shape)

if __name__ == "__main__":
    unittest.main()
