import argparse
import logging
import os

import pandas as pd

from DataPreparation import DataPreparation


def setup_logging():
    logging.basicConfig(
        filename="data_preparation.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def clean_and_generate_report(df, schema_path, y_label):
    """
    Automatically cleans the dataset in a pre-defined order.
    Note: this order chan be changed as per the data cleaning use-case
    Report creation and cleaning go hand in hand.
    """

    # Create a new instance of the DataPreparation class for assessment
    dp_obj = DataPreparation(df, y_label)

    dp_obj.detect_schema_violations(schema_path)
    dp_obj.remove_duplicates()
    dp_obj.fill_missing_values()
    dp_obj.remove_outliers()
    dp_obj.remove_correlated_metrics()
    dp_obj.remove_redundant_metrics()
    dp_obj.class_overlap_removal()

    # TODO: update this to use reportlab and generate pdf report.
    report_file = "data_quality_report.txt"
    with open(report_file, "w") as f:
        f.write(open("data_preparation.log", "r").read())

    logging.info(f"Report saved at {report_file}")
    cleaned_df = dp_obj.df
    return cleaned_df


def main():
    parser = argparse.ArgumentParser(description="Data Quality Assessment Tool")
    parser.add_argument("dataset", type=str, help="Path to the dataset in CSV format")
    parser.add_argument(
        "--schema", type=str, default=None, help="Optional path to the schema JSON file"
    )
    parser.add_argument(
        "--y_label",
        type=str,
        default=None,
        help="Optional flag to auto-clean the issues found",
    )

    args = parser.parse_args()
    dataset_path = args.dataset
    schema_path = args.schema
    y_label = args.y_label

    # Checking if the dataset is valid csv file!
    if not os.path.exists(schema_path):
        raise ValueError("Schema path does not exist.")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError("Error reading dataframe with Error: %s" % e)

    clean_and_generate_report(df, schema_path, y_label)


if __name__ == "__main__":
    setup_logging()
    main()
