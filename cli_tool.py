import argparse
import logging
import os, datetime

import pandas as pd
from log_to_pdf import create_pdf
from DataPreparation import DataPreparation


def do_all_cleaning(df, schema_path, y_label):
    """
    Automatically cleans the dataset in a pre-defined order.
    Note: this order chan be changed as per the data cleaning use-case
    Report creation and cleaning go hand in hand.
    """

    # Create a new instance of the DataPreparation class for assessment
    dp_obj = DataPreparation(df, y_label)

    dp_obj.report_schema_violations(schema_path)
    dp_obj.remove_duplicates()
    dp_obj.fill_missing_values()
    dp_obj.report_differently_distributed_feature()
    dp_obj.remove_outliers()
    dp_obj.remove_constant_features()
    dp_obj.remove_correlated_metrics()
    dp_obj.remove_redundant_metrics()
    dp_obj.class_overlap_removal()

    return dp_obj.df


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


    _id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_file_name = f"logs/data_cleaning_{os.path.basename(dataset_path).split('.')[0]}_{_id}.log"
    # Initialize the logging based on the unique name generated
    logging.basicConfig(
        filename=log_file_name,
        level=logging.INFO,
        # format="%(asctime)s - %(levelname)s - %(message)s",
        format="%(message)s"
    )

    # Checking if the dataset is valid csv file!
    if not os.path.exists(dataset_path):
        raise ValueError(f"{dataset_path=} does not exist.")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise ValueError("Error reading dataframe with Error: %s" % e)

    cleaned_df = do_all_cleaning(df, schema_path, y_label)

    # saving log as pdf for report
    report_name = f"CleaningOutputs/Report_{os.path.basename(dataset_path).split('.')[0]}_{_id}.pdf"
    create_pdf(log_filename = log_file_name, output_filename = report_name)

    # saving the cleaned data
    cleaned_name = f"CleaningOutputs/Cleaned_{os.path.basename(dataset_path).split('.')[0]}_{_id}.csv"
    cleaned_df.to_csv(cleaned_name, index=False)


if __name__ == "__main__":
    main()
