# Data Quality Assessment & Cleaning Tool

A command-line tool that provides data quality assessment, schema violation detection, auto cleaning, and logging capabilities for tabular data.

## Features
*Data Quality Assessment*: Evaluates datasets for common antipatterns.

*Schema Violation Detection*: Compares dataset against a provided JSON schema and identifies violations.

*Auto Cleaning Suggestions*: Offers recommendations for cleaning datasets based on identified issues.

*Logging Capabilities*: Logs each step of the process for auditing and debugging purposes.

*Unit Tests*: Comprehensive unit tests to ensure reliability and correctness.

## Directory Structure
```
.
├── cli_tool.py
├── DataPreparation.py
└── unittests
    └── test_schema_violations.py
```

## Getting Started

### Prerequisites:

- Python 3.x
- pandas (`pip install pandas`)

### Usage:

1. Run the CLI tool using:
   
```bash
python cli_tool.py <path_to_dataset.csv> --schema <path_to_schema.json>
```

For Example:
```bash
python cli_tool.py dataset.csv --schema schema.json
```
After running the tool, it will generate:

A log file named `data_preparation.log` detailing the assessment and any issues found.
A cleaned dataset named `cleaned_dataset.csv`.

##### Running Tests:
Navigate to the unittests dir and run:

```
python -m unittest test_schema_violations.py
```