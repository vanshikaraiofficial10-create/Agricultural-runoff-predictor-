import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("\n--------------------------------------------")
print(" AGRICULTURAL RUNOFF PREDICTION SYSTEM (AI)")
print("--------------------------------------------\n")

# ----------------------------------------------------------
# 1. CREATE REALISTIC SAMPLE DATASET
# ----------------------------------------------------------

np.random.seed(42)
rows = 350

data = {
    "rainfall": np.random.randint(10, 300, rows),       # mm
    "soil_type": np.random.choice(["sandy", "clay", "loam"], rows),
    "fertilizer": np.random.uniform(10, 120, rows),     # kg/ha
    "pesticide": np.random.uniform(0.1, 10, rows),      # mg/L
    "distance_to_water": np.random.randint(5, 300, rows),  # meters
    "soil_moisture": np.random.uniform(5, 50, rows),    # %
    "nitrate": np.random.uniform(1, 50, rows),          # mg/L
    "phosphate": np.random.uniform(0.5, 30, rows)       # mg/L
}

df = pd.DataFrame(data)

# ----------------------------------------------------------
# 2. CREATE TARGET LABEL (Safe / Moderate Risk / High Risk)
# ----------------------------------------------------------

def label_risk(row):
    score = 0
    
    if row["rainfall"] > 180: score += 1
    if row["soil_type"] == "sandy": score += 1
    if row["fertilizer"] > 80: score += 1
    if row["pesticide"] > 4: score += 1
    if row["nitrate"] > 25: score += 1
    if row["phosphate"] > 12: score += 1
    if row["distance_to_water"] < 50: score += 1

    if score <= 2:
        return "Safe"
    elif score <= 4:
        return "Moderate Risk"
    else:
        return "High Risk"

df["status"] = df.apply(label_risk, axis=1)

# ----------------------------------------------------------
# 3. ENCODE CATEGORICAL DATA (soil type)
# ----------------------------------------------------------

encoder = LabelEncoder()
df["soil_type"] = encoder.fit_transform(df["soil_type"])

# ----------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ----------------------------------------------------------

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------
# 5. TRAIN MODEL (Random Forest)
# ----------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------------------------------
# 6. EVALUATION
# ----------------------------------------------------------

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Model Accuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, preds))

print("\nModel trained successfully!\n")

# ----------------------------------------------------------
# 7. USER INPUT PREDICTION (CLEAN INPUT)
# ----------------------------------------------------------

print("\n----------- PREDICT RUNOFF CONDITION -----------")

try:
    rainfall = float(input("Enter Rainfall (in mm): "))
    soil_type_input = input("Soil Type (sandy / clay / loam): ").lower()
    fertilizer = float(input("Fertilizer used (kg/ha): "))
    pesticide = float(input("Pesticide level (mg/L): "))
    distance = float(input("Distance to nearest water body (m): "))
    moisture = float(input("Soil moisture (%): "))
    nitrate = float(input("Nitrate level (mg/L): "))
    phosphate = float(input("Phosphate level (mg/L): "))

    if soil_type_input not in ["sandy", "clay", "loam"]:
        print("\nInvalid soil type given! Please enter sandy / clay / loam.")
        exit()

    soil_encoded = encoder.transform([soil_type_input])[0]

    new_data = [[rainfall, soil_encoded, fertilizer, pesticide,
                 distance, moisture, nitrate, phosphate]]

    prediction = model.predict(new_data)[0]

    print("\n-----------------------------------------------")
    print(" Predicted Water Status:", prediction)
    print("-----------------------------------------------\n")

except Exception as e:
    print("\nError:", e)
