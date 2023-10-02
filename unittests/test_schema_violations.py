import unittest

import numpy as np
import pandas as pd

from DataPreparation import DataPreparation


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
        violations = dp_obj.detect_schema_violations(schema_path="sample_schema.json")
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
        # TODO: fix missing values first hten this test will work.
        dp_obj = DataPreparation(df=self.df, y_label='RealBug')  # assuming C as target column
        # for class overlap to work, first we need to remove Nas in teh dataset
        _ = dp_obj.fill_missing_values()
        overlapped_row_count = dp_obj.class_overlap_removal()
        print(overlapped_row_count)


# TODO: finally do end-to-end testing on shell
# need to test main.py -- test case for that? as gpt!
# update the log files

if __name__ == "__main__":
    unittest.main()
