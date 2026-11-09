
from pyspark.sql import SparkSession

# ==========================================
# 1. INITIALIZE SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("TelecomCustomerChurnAnalytics")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("=" * 50)
print("TELECOM CUSTOMER CHURN ANALYTICS")
print("SPARK DATA INGESTION")
print("=" * 50)


# ==========================================
# 2. DEFINE DATASET PATHS
# ==========================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"

dataset_paths = {
    "customers": RAW_DIR / "customers.csv",
    "subscriptions": RAW_DIR / "subscriptions.csv",
    "usage": RAW_DIR / "usage.csv",
    "recharge": RAW_DIR / "recharge.csv",
    "complaints": RAW_DIR / "complaints.csv",
    "churn": RAW_DIR / "churn.csv"
}


# ==========================================
# 3. LOAD CSV FILES INTO SPARK DATAFRAMES
# ==========================================

dataframes = {}

for name, path in dataset_paths.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "FAILFAST")
        .csv(str(path))
    )

    dataframes[name] = df

    print(f"\nLoaded dataset: {name}")
    print(f"File: {path.name}")


# ==========================================
# 4. DISPLAY DATASET SCHEMAS
# ==========================================

for name, df in dataframes.items():

    print("\n" + "=" * 50)
    print(f"SCHEMA: {name.upper()}")
    print("=" * 50)

    df.printSchema()


# ==========================================
# 5. DISPLAY SAMPLE RECORDS
# ==========================================

for name, df in dataframes.items():

    print("\n" + "=" * 50)
    print(f"SAMPLE RECORDS: {name.upper()}")
    print("=" * 50)

    df.show(5, truncate=False)


# ==========================================
# 6. COUNT DATASET RECORDS
# ==========================================

print("\n" + "=" * 50)
print("DATASET RECORD SUMMARY")
print("=" * 50)

for name, df in dataframes.items():

    print(f"{name}: {df.count()} records")


# ==========================================
# 7. BASIC DATA VALIDATION
# ==========================================

print("\n" + "=" * 50)
print("DATA VALIDATION")
print("=" * 50)

required_columns = {
    "customers": ["customer_id"],
    "subscriptions": ["customer_id"],
    "usage": ["customer_id", "usage_date"],
    "recharge": ["customer_id", "recharge_id"],
    "complaints": ["customer_id", "complaint_id"],
    "churn": ["customer_id", "churn_status"]
}

for name, columns in required_columns.items():

    df = dataframes[name]

    missing_columns = [
        col for col in columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: {missing_columns}"
        )

    print(f"{name}: Required columns verified")


# ==========================================
# 8. CHECK CUSTOMER ID NULL VALUES
# ==========================================

from pyspark.sql.functions import col

for name, df in dataframes.items():

    if "customer_id" in df.columns:

        null_count = (
            df.filter(col("customer_id").isNull())
            .count()
        )

        print(
            f"{name}: Null customer IDs = {null_count}"
        )


# ==========================================
# 9. STORE DATAFRAMES FOR LATER USE
# ==========================================

customers_df = dataframes["customers"]
subscriptions_df = dataframes["subscriptions"]
usage_df = dataframes["usage"]
recharge_df = dataframes["recharge"]
complaints_df = dataframes["complaints"]
churn_df = dataframes["churn"]


# ==========================================
# 10. FINISH
# ==========================================

print("\n" + "=" * 50)
print("ALL DATASETS INGESTED SUCCESSFULLY!")
print("=" * 50)

spark.stop()
