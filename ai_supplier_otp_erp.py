# ===============================================
# AI ENABLED SUPPLIER ON-TIME PERFORMANCE SYSTEM
# ===============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# -------------------------------
# 1. MASTER DATA CONFIGURATION
# -------------------------------

np.random.seed(42)

VENDORS = ["VENDOR_A", "VENDOR_B", "VENDOR_C", "VENDOR_D"]
FACILITIES = ["FACILITY_DAY", "FACILITY_NIGHT"]
BUYERS = ["BUYER_1", "BUYER_2"]

vendor_lead_time = {
    "VENDOR_A": 5,
    "VENDOR_B": 7,
    "VENDOR_C": 4,
    "VENDOR_D": 6
}

night_facilities = ["FACILITY_NIGHT"]

vendor_exclusion = {"VENDOR_D": False}
buyer_exclusion = {"BUYER_2": False}
facility_exclusion = {"FACILITY_DAY": False}

# -------------------------------
# 2. GENERATE SIMULATED PO DATA
# -------------------------------

def generate_po_data(num_records=500):
    data = []
    for i in range(num_records):
        vendor = random.choice(VENDORS)
        facility = random.choice(FACILITIES)
        buyer = random.choice(BUYERS)

        order_date = datetime(2025, 1, 1) + timedelta(days=random.randint(0, 60))
        lead_time = vendor_lead_time[vendor]
        due_date = order_date + timedelta(days=lead_time)

        delay = random.randint(-1, 3)
        received_date = due_date + timedelta(days=delay)

        received_time = random.randint(6, 18)

        ordered_qty = random.randint(100, 500)
        received_qty = ordered_qty - random.randint(0, 50)

        data.append([
            i, vendor, facility, buyer,
            order_date, due_date, received_date,
            received_time, lead_time,
            ordered_qty, received_qty
        ])

    columns = [
        "PO_ID", "Vendor", "Facility", "Buyer",
        "Order_Date", "Due_Date", "Received_Date",
        "Received_Hour", "Lead_Time",
        "Ordered_Qty", "Received_Qty"
    ]

    return pd.DataFrame(data, columns=columns)

df = generate_po_data()

# -------------------------------
# 3. EXCLUSION ENGINE
# -------------------------------

def apply_exclusions(df):
    df["Excluded"] = False

    for index, row in df.iterrows():
        if vendor_exclusion.get(row["Vendor"], False):
            df.at[index, "Excluded"] = True
        if buyer_exclusion.get(row["Buyer"], False):
            df.at[index, "Excluded"] = True
        if facility_exclusion.get(row["Facility"], False):
            df.at[index, "Excluded"] = True

    return df

df = apply_exclusions(df)

# -------------------------------
# 4. OTP CALCULATION ENGINE
# -------------------------------

def calculate_otp(df):
    df["Days_Difference"] = (df["Received_Date"] - df["Due_Date"]).dt.days
    df["Order_Diff"] = (df["Received_Date"] - df["Order_Date"]).dt.days

    def is_late(row):
        if row["Excluded"]:
            return False

        # Non-night logic
        if row["Facility"] not in night_facilities:
            if row["Days_Difference"] >= 1 and row["Order_Diff"] > row["Lead_Time"]:
                return True

        # Night logic
        else:
            if (row["Days_Difference"] >= 1 and
                row["Order_Diff"] > row["Lead_Time"] and
                row["Received_Hour"] > 12):
                return True

        return False

    df["Is_Late"] = df.apply(is_late, axis=1)
    return df

df = calculate_otp(df)

# -------------------------------
# 5. PERFORMANCE METRICS
# -------------------------------

df["Fill_Rate"] = np.minimum(df["Received_Qty"] / df["Ordered_Qty"], 1.0)

total_pos = len(df[~df["Excluded"]])
late_pos = len(df[(df["Is_Late"]) & (~df["Excluded"])])

otp_percent = 100 - (late_pos / total_pos * 100)

# Fine Logic
if otp_percent <= 90:
    fine = late_pos * 500
elif otp_percent < 95:
    fine = late_pos * 300
else:
    fine = 0

print("\n==============================")
print("OVERALL OTP PERFORMANCE")
print("==============================")
print(f"Total Eligible POs: {total_pos}")
print(f"Late POs: {late_pos}")
print(f"OTP %: {otp_percent:.2f}")
print(f"Total Fine: ${fine}")
print("==============================\n")

# -------------------------------
# 6. AI LATE PREDICTION MODEL
# -------------------------------

model_df = df[~df["Excluded"]].copy()

features = model_df[[
    "Lead_Time", "Ordered_Qty", "Received_Hour", "Days_Difference"
]]

target = model_df["Is_Late"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("AI Late Prediction Model Performance:")
print(classification_report(y_test, predictions))

# -------------------------------
# 7. AI DISPUTE RECOMMENDATION ENGINE
# -------------------------------

def dispute_recommendation(row):
    probability = model.predict_proba([[
        row["Lead_Time"],
        row["Ordered_Qty"],
        row["Received_Hour"],
        row["Days_Difference"]
    ]])[0][1]

    if probability > 0.80:
        return "Reject Dispute - High Confidence Late"
    elif probability > 0.50:
        return "Review Manually"
    else:
        return "Approve Dispute - Likely Justified"

df["AI_Dispute_Recommendation"] = df.apply(dispute_recommendation, axis=1)

# -------------------------------
# 8. SUPPLIER SCORECARD
# -------------------------------

scorecard = df[~df["Excluded"]].groupby("Vendor").agg({
    "Is_Late": "sum",
    "PO_ID": "count",
    "Fill_Rate": "mean"
}).reset_index()

scorecard["OTP_%"] = 100 - (scorecard["Is_Late"] / scorecard["PO_ID"] * 100)

print("\nSUPPLIER SCORECARD")
print(scorecard)

# -------------------------------
# 9. EXECUTIVE AI SUMMARY
# -------------------------------

def generate_summary():
    worst_vendor = scorecard.sort_values("OTP_%").iloc[0]
    best_vendor = scorecard.sort_values("OTP_%", ascending=False).iloc[0]

    summary = f"""
Executive Summary:

Overall OTP stands at {otp_percent:.2f}%.
Total financial exposure from fines is ${fine}.

Top Performing Vendor: {best_vendor['Vendor']} with OTP {best_vendor['OTP_%']:.2f}%.
Lowest Performing Vendor: {worst_vendor['Vendor']} with OTP {worst_vendor['OTP_%']:.2f}%.

AI indicates predictive patterns based on lead time deviation and receiving hour variance.
Recommended focus: Improve coordination with low-performing vendors and optimize lead time adherence.
"""
    return summary

print(generate_summary())

# -------------------------------
# 10. REAL-TIME DASHBOARD SIMULATION
# -------------------------------

plt.figure()
scorecard.plot(x="Vendor", y="OTP_%", kind="bar")
plt.title("Supplier OTP Performance")
plt.ylabel("OTP %")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 11. AUDIT TRAIL SIMULATION
# -------------------------------

df["Audit_Timestamp"] = datetime.now()
df["Audit_Action"] = "OTP Calculated & AI Validated"

print("\nSystem Audit Snapshot:")
print(df[["PO_ID", "Vendor", "Is_Late", "AI_Dispute_Recommendation"]].head())

print("\nAI ENABLED SUPPLIER OTP PROCESS COMPLETED SUCCESSFULLY.")