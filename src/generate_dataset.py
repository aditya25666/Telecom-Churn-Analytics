
import csv
import random
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------
# CONFIGURATION
# ---------------------------------

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_CUSTOMERS = 10000
TODAY = date(2026, 10, 31)
ANALYSIS_START = date(2025, 11, 1)

CITIES = [
    "Mumbai", "Pune", "Navi Mumbai", "Delhi",
    "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Ahmedabad", "Jaipur"
]

PLAN_TYPES = ["Prepaid", "Postpaid"]
CONTRACT_TYPES = ["Monthly", "One Year", "Two Years"]
PAYMENT_METHODS = ["UPI", "Debit Card", "Credit Card", "Net Banking"]
COMPLAINT_TYPES = [
    "Network Issue",
    "Billing Issue",
    "Internet Issue",
    "Call Drop",
    "Recharge Failure",
    "Customer Support"
]

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def random_date(start, end):
    if start >= end:
        return start

    days = (end - start).days
    return start + timedelta(days=random.randint(0, days))


def write_csv(filename, columns, rows):
    filepath = RAW_DIR / filename

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {filename}: {len(rows)} records")


# ---------------------------------
# DATA STORAGE
# ---------------------------------

customers = []
subscriptions = []
usage_records = []
recharge_records = []
complaint_records = []
churn_records = []

# ---------------------------------
# GENERATE CUSTOMER DATA
# ---------------------------------

for i in range(1, TOTAL_CUSTOMERS + 1):

    customer_id = f"CUST{i:05d}"

    joining_date = random_date(
        date(2022, 1, 1),
        date(2026, 1, 1)
    )

    tenure = (
        (TODAY.year - joining_date.year) * 12
        + TODAY.month - joining_date.month
    )

    gender = random.choice(["Male", "Female"])
    age_group = random.choice([
        "18-25", "26-35", "36-45", "46-60", "60+"
    ])

    city = random.choice(CITIES)

    customers.append({
        "customer_id": customer_id,
        "gender": gender,
        "age_group": age_group,
        "city": city,
        "joining_date": joining_date.isoformat()
    })

    # ---------------------------------
    # SUBSCRIPTION DATA
    # ---------------------------------

    plan_type = random.choice(PLAN_TYPES)

    contract_type = random.choices(
        CONTRACT_TYPES,
        weights=[60, 25, 15]
    )[0]

    monthly_charges = round(
        random.uniform(199, 1299), 2
    )

    internet_service = random.choice([
        "Fiber", "DSL", "No Internet"
    ])

    subscriptions.append({
        "customer_id": customer_id,
        "plan_type": plan_type,
        "contract_type": contract_type,
        "monthly_charges": monthly_charges,
        "internet_service": internet_service
    })

    # ---------------------------------
    # RECHARGE DATA
    # ---------------------------------

    recharge_count = random.randint(5, 9)
    customer_recharges = []

    for j in range(recharge_count):

        recharge_date = random_date(
            ANALYSIS_START, TODAY
        )

        amount = random.choice([
            199, 249, 299, 399, 499, 599, 799, 999
        ])

        customer_recharges.append({
            "recharge_id": f"R{i:05d}_{j+1:02d}",
            "customer_id": customer_id,
            "recharge_date": recharge_date.isoformat(),
            "recharge_amount": amount,
            "payment_method": random.choice(PAYMENT_METHODS)
        })

    # ---------------------------------
    # COMPLAINT DATA
    # ---------------------------------

    complaint_count = random.choices(
        [0, 1, 2, 3, 4],
        weights=[45, 30, 15, 7, 3]
    )[0]

    customer_complaints = []

    for j in range(complaint_count):

        complaint_date = random_date(
            ANALYSIS_START, TODAY
        )

        resolved = random.choice([True, False])

        customer_complaints.append({
            "complaint_id": f"COMP{i:05d}_{j+1:02d}",
            "customer_id": customer_id,
            "complaint_type": random.choice(COMPLAINT_TYPES),
            "complaint_date": complaint_date.isoformat(),
            "resolution_status": (
                "Resolved" if resolved else "Pending"
            ),
            "resolution_days": (
                random.randint(1, 7) if resolved else ""
            )
        })

    # ---------------------------------
    # CHURN STATUS
    # ---------------------------------

    # This is a synthetic business-rule
    # based label, NOT an ML prediction.

    churn_probability = 0.08

    if contract_type == "Monthly":
        churn_probability += 0.12

    if tenure < 12:
        churn_probability += 0.12

    if complaint_count >= 2:
        churn_probability += 0.10

    if recharge_count <= 5:
        churn_probability += 0.06

    churn_probability = min(churn_probability, 0.65)

    churned = random.random() < churn_probability

    churn_date = ""

    if churned:
        churn_date = random_date(
            date(2026, 8, 1), TODAY
        )

    churn_records.append({
        "customer_id": customer_id,
        "churn_status": "Yes" if churned else "No",
        "churn_date": churn_date.isoformat() if churned else ""
    })

    # ---------------------------------
    # USAGE DATA
    # ---------------------------------

    # Generate one usage record per month
    # for up to six recent months.

    for month_offset in range(min(6, max(1, tenure))):

        month_date = TODAY - timedelta(
            days=30 * month_offset
        )

        usage_date = date(
            month_date.year,
            month_date.month,
            1
        )

        if churned and usage_date > churn_date:
            continue

        if usage_date < joining_date:
            continue

        usage_records.append({
            "customer_id": customer_id,
            "usage_date": usage_date.isoformat(),
            "data_usage_gb": round(
                random.uniform(1, 80), 2
            ),
            "call_minutes": random.randint(20, 1500),
            "sms_count": random.randint(0, 500)
        })

    # ---------------------------------
    # FILTER EVENTS AFTER CHURN
    # ---------------------------------

    if churned:
        recharge_records.extend([
            r for r in customer_recharges
            if date.fromisoformat(r["recharge_date"]) <= churn_date
        ])

        complaint_records.extend([
            c for c in customer_complaints
            if date.fromisoformat(c["complaint_date"]) <= churn_date
        ])
    else:
        recharge_records.extend(customer_recharges)
        complaint_records.extend(customer_complaints)


# ---------------------------------
# SAVE ALL SIX DATASETS
# ---------------------------------

write_csv(
    "customers.csv",
    ["customer_id", "gender", "age_group", "city", "joining_date"],
    customers
)

write_csv(
    "subscriptions.csv",
    ["customer_id", "plan_type", "contract_type",
     "monthly_charges", "internet_service"],
    subscriptions
)

write_csv(
    "usage.csv",
    ["customer_id", "usage_date", "data_usage_gb",
     "call_minutes", "sms_count"],
    usage_records
)

write_csv(
    "recharge.csv",
    ["recharge_id", "customer_id", "recharge_date",
     "recharge_amount", "payment_method"],
    recharge_records
)

write_csv(
    "complaints.csv",
    ["complaint_id", "customer_id", "complaint_type",
     "complaint_date", "resolution_status", "resolution_days"],
    complaint_records
)

write_csv(
    "churn.csv",
    ["customer_id", "churn_status", "churn_date"],
    churn_records
)

print("\nAll telecom datasets generated successfully!")
print("Dataset location:", RAW_DIR)
