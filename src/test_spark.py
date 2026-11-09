
from pyspark.sql import SparkSession

# Create Spark session
spark = (
    SparkSession.builder
    .appName("TelecomChurnAnalytics")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

# Display Spark information
print("Spark Started Successfully!")
print("Spark Version:", spark.version)

# Create sample telecom customer data
data = [
    (101, "Prepaid", 299, 12),
    (102, "Postpaid", 599, 24),
    (103, "Prepaid", 199, 6),
    (104, "Postpaid", 799, 36),
    (105, "Prepaid", 399, 18)
]

columns = [
    "customer_id",
    "plan_type",
    "monthly_charge",
    "tenure"
]

df = spark.createDataFrame(data, columns)

# Display customer records
df.show()

# Perform basic analytics
print("Average Monthly Charge:")

df.selectExpr(
    "AVG(monthly_charge) AS average_charge"
).show()

# Stop Spark
spark.stop()
