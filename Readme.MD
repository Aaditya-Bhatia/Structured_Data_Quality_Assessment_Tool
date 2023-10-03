# Data Quality Assessment & Cleaning Tool

## Overview
This tool is designed to automate the data cleaning process for a given tabular dataset. It provides comprehensive cleaning steps including handling duplicates, missing values, outlier detection, and more. After the process, a data quality report is generated in PDF format along with the cleaned dataset.


## Features
*Data Quality Assessment*: Evaluates datasets for common data quality issues. The issues are from a ML model training standpoint. 

*Schema Violation Detection*: Compares dataset against a provided JSON schema and identifies violations.

*Class Overlap Detection*: Uses the response variable to check if class overlap is present in dataset. Class overlap is harmful data quality issue for classification tasks.

*Auto Cleaning Suggestions*: Offers recommendations for cleaning datasets based on identified issues.

*Report Generation*: Logs all data quality issues into a pdf report.

*Unit Tests*: Comprehensive unit tests to ensure reliability and correctness.


## Getting Started

### Prerequisites:

- Python 3.x
- pandas (`pip install -r requirements.txt`)

### Usage:

1. Run the CLI tool using:
   
```bash
python cli_tool.py <path_to_dataset.csv> --schema <path_to_schema.json> --y_label <response label for dataset>
```

For Example:
```bash
python cli_tool.py dataset.csv --schema schema.json --y_label IS_CAT_OR_DOG
```
After running the tool, it will generate:

## Outputs
The tool generates the following outputs:

1. **Logs:**

    Folder: `logs`
 
    Naming Format: `data_cleaning_<dataset_name>_<timestamp>.log`

    Description: This log captures all the cleaning steps undertaken by the tool, including information about what was changed and any detected anomalies. It provides insights into the quality of the data and any modifications made.


2. **Data Quality Report:**

    Folder: `CleaningOutputs/`

    Naming Format: `Report_<dataset_name>_<timestamp>.pdf`

    Description: A comprehensive report detailing the quality of the data and the transformations applied. It's generated from the logs and is formatted for clear presentation.


3. Cleaned Dataset:

    Folder: `CleaningOutputs/`
    
    Naming Format: `Cleaned_<dataset_name>_<timestamp>.csv`
    
    Description: The cleaned version of the input dataset. All the identified issues in the data have been rectified in this dataset.

#### Running Tests:
Navigate to the unittests dir and run:

```
python -m unittest tests/unittests.py
```