
"""
TELECOM CUSTOMER CHURN ANALYTICS
Phase 4: Data Cleaning and Preprocessing

Processing:
1. Read six raw CSV datasets using PySpark.
2. Validate expected columns.
3. Clean string values and convert data types.
4. Remove duplicate rows.
5. Check missing values.
6. Save cleaned datasets as Parquet using Pandas + PyArrow.

Note:
PySpark performs the data processing.
Pandas + PyArrow are used only for writing Parquet files
to avoid the Windows winutils.exe filesystem issue.
"""

from pathlib import Path
import shutil
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    DoubleType,
    IntegerType
)


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

# Project root = parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "customers": {
        "file": "customers.csv",
        "required_columns": [
            "customer_id",
            "gender",
            "age_group",
            "city",
            "joining_date"
        ],
        "date_columns": [
            "joining_date"
        ],
        "numeric_columns": {}
    },

    "subscriptions": {
        "file": "subscriptions.csv",
        "required_columns": [
            "customer_id",
            "plan_type",
            "contract_type",
            "monthly_charges",
            "internet_service"
        ],
        "date_columns": [],
        "numeric_columns": {
            "monthly_charges": DoubleType()
        }
    },

    "usage": {
        "file": "usage.csv",
        "required_columns": [
            "customer_id",
            "usage_date",
            "data_usage_gb",
            "call_minutes",
            "sms_count"
        ],
        "date_columns": [
            "usage_date"
        ],
        "numeric_columns": {
            "data_usage_gb": DoubleType(),
            "call_minutes": DoubleType(),
            "sms_count": IntegerType()
        }
    },

    "recharge": {
        "file": "recharge.csv",
        "required_columns": [
            "recharge_id",
            "customer_id",
            "recharge_date",
            "recharge_amount",
            "payment_method"
        ],
        "date_columns": [
            "recharge_date"
        ],
        "numeric_columns": {
            "recharge_amount": DoubleType()
        }
    },

    "complaints": {
        "file": "complaints.csv",
        "required_columns": [
            "complaint_id",
            "customer_id",
            "complaint_type",
            "complaint_date",
            "resolution_status",
            "resolution_days"
        ],
        "date_columns": [
            "complaint_date"
        ],
        "numeric_columns": {
            "resolution_days": IntegerType()
        }
    },

    "churn": {
        "file": "churn.csv",
        "required_columns": [
            "customer_id",
            "churn_status",
            "churn_date"
        ],
        "date_columns": [
            "churn_date"
        ],
        "numeric_columns": {}
    }
}


# ============================================================
# 3. CREATE SPARK SESSION
# ============================================================

def create_spark_session():

    print("\nInitializing Apache Spark...")

    spark = (
        SparkSession.builder
        .appName("TelecomCustomerChurnAnalytics")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config(
            "spark.sql.execution.arrow.pyspark.enabled",
            "false"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("Spark initialized successfully.")
    print("Spark Version:", spark.version)

    return spark


# ============================================================
# 4. READ RAW CSV DATASET
# ============================================================

def read_dataset(spark, dataset_name, config):

    file_path = RAW_DIR / config["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print(f"\nReading {dataset_name}.csv ...")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(str(file_path))
    )

    print(f"{dataset_name} loaded successfully.")

    return df


# ============================================================
# 5. VALIDATE DATASET COLUMNS
# ============================================================

def validate_columns(df, dataset_name, required_columns):

    actual_columns = df.columns

    missing_columns = [
        column
        for column in required_columns
        if column not in actual_columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    print(f"{dataset_name}: Column validation passed.")


# ============================================================
# 6. CLEAN STRING COLUMNS
# ============================================================

def clean_string_columns(df):

    for column_name, data_type in df.dtypes:

        if data_type == "string":

            df = df.withColumn(
                column_name,
                F.trim(F.col(column_name))
            )

            # Convert empty strings into NULL.
            df = df.withColumn(
                column_name,
                F.when(
                    F.col(column_name) == "",
                    None
                ).otherwise(F.col(column_name))
            )

    return df


# ============================================================
# 7. CONVERT DATA TYPES
# ============================================================

def convert_data_types(df, config):

    # Convert date columns.
    for column_name in config["date_columns"]:

        df = df.withColumn(
            column_name,
            F.to_date(F.col(column_name))
        )

    # Convert numeric columns.
    for column_name, target_type in (
        config["numeric_columns"].items()
    ):

        df = df.withColumn(
            column_name,
            F.col(column_name).cast(target_type)
        )

    return df


# ============================================================
# 8. REMOVE DUPLICATE RECORDS
# ============================================================

def remove_duplicates(df, dataset_name):

    before_count = df.count()

    df = df.dropDuplicates()

    after_count = df.count()

    removed_count = before_count - after_count

    print(f"\n{dataset_name} duplicate check:")
    print("Records before cleaning:", before_count)
    print("Records after cleaning:", after_count)
    print("Duplicate rows removed:", removed_count)

    return df


# ============================================================
# 9. DATA QUALITY SUMMARY
# ============================================================

def display_data_quality(df, dataset_name):

    print("\n" + "=" * 60)
    print(f"DATA QUALITY REPORT: {dataset_name.upper()}")
    print("=" * 60)

    total_records = df.count()

    print("Total Records:", total_records)

    print("\nDataset Schema:")
    df.printSchema()

    print("\nSample Records:")
    df.show(5, truncate=False)

    # Calculate NULL values column-wise.
    null_expressions = [
        F.sum(
            F.when(F.col(column).isNull(), 1)
            .otherwise(0)
        ).alias(column)
        for column in df.columns
    ]

    print("\nMissing Value Count:")
    df.select(null_expressions).show(truncate=False)


# ============================================================
# 10. SAVE DATASET AS PARQUET USING PYARROW
# ============================================================

def save_as_parquet(df, dataset_name):

    output_path = PROCESSED_DIR / dataset_name

    # Remove incomplete output from previous failed runs.
    if output_path.exists():

        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nSaving {dataset_name} as Parquet...")

    # Collect the cleaned Spark DataFrame into Pandas.
    pandas_df = df.toPandas()

    # Save using PyArrow instead of Spark's Hadoop writer.
    parquet_file = output_path / "data.parquet"

    pandas_df.to_parquet(
        parquet_file,
        engine="pyarrow",
        index=False
    )

    print("Parquet saved successfully:")
    print(parquet_file)

    # Release Pandas memory.
    del pandas_df


# ============================================================
# 11. PROCESS INDIVIDUAL DATASET
# ============================================================

def process_dataset(spark, dataset_name, config):

    print("\n")
    print("#" * 65)
    print(f"PROCESSING DATASET: {dataset_name.upper()}")
    print("#" * 65)

    # Step 1: Read raw CSV.
    df = read_dataset(
        spark,
        dataset_name,
        config
    )

    # Step 2: Validate expected columns.
    validate_columns(
        df,
        dataset_name,
        config["required_columns"]
    )

    # Step 3: Clean strings.
    df = clean_string_columns(df)

    # Step 4: Convert data types.
    df = convert_data_types(
        df,
        config
    )

    # Step 5: Remove exact duplicate rows.
    df = remove_duplicates(
        df,
        dataset_name
    )

    # Step 6: Display quality report.
    display_data_quality(
        df,
        dataset_name
    )

    # Step 7: Save cleaned data.
    save_as_parquet(
        df,
        dataset_name
    )

    print(
        f"\n{dataset_name.upper()} processing completed."
    )

    return df.count()


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("TELECOM CUSTOMER CHURN ANALYTICS")
    print("PHASE 4: DATA CLEANING AND PREPROCESSING")
    print("=" * 65)

    print("\nProject directory:", PROJECT_ROOT)
    print("Raw dataset directory:", RAW_DIR)
    print("Processed dataset directory:", PROCESSED_DIR)

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DIR}"
        )

    spark = None

    processing_summary = {}

    try:

        # Initialize Spark.
        spark = create_spark_session()

        # Process all six datasets.
        for dataset_name, config in DATASETS.items():

            record_count = process_dataset(
                spark,
                dataset_name,
                config
            )

            processing_summary[dataset_name] = record_count

        # Final summary.
        print("\n")
        print("=" * 65)
        print("FINAL DATA CLEANING SUMMARY")
        print("=" * 65)

        for dataset_name, count in processing_summary.items():

            print(
                f"{dataset_name:<20} : {count:>10,} records"
            )

        print("\nAll datasets processed successfully.")

        print("\nProcessed files are available at:")
        print(PROCESSED_DIR)

    except Exception as error:

        print("\nDATA CLEANING FAILED!")
        print("Error:", str(error))

        raise

    finally:

        if spark is not None:

            spark.stop()

            print("\nSpark session stopped.")


# ============================================================
# 13. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
