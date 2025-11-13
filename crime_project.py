import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("Crime_Reports_20240701.csv")
print("\nDataset Loaded")
print(df.head())
print("Shape:", df.shape)

# 2. Cleaning
df = df.dropna(subset=["Crime Date Time"])
df["Crime Date Time"] = pd.to_datetime(df["Crime Date Time"], errors="coerce")
df = df.dropna(subset=["Crime Date Time"])

print("\nCleaned missing date/time rows")
print("New shape:", df.shape)

# 3. Feature Engineering
df["Hour"] = df["Crime Date Time"].dt.hour
df["Day"] = df["Crime Date Time"].dt.day_name()
df["Month"] = df["Crime Date Time"].dt.month

print("\n Added Hour, Day, Month columns")

# 4. Crimes per hour (EDA)
hourly = df.groupby("Hour").size()
print("\n Crimes per Hour:")
print(hourly)

plt.figure()
plt.bar(hourly.index, hourly.values)
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Crimes")
plt.title("Crimes by Hour")
plt.show()

# 5. Crimes per Day
daywise = df.groupby("Day").size().sort_values(ascending=False)
print("\n Crimes per Day:")
print(daywise)

plt.figure()
plt.bar(daywise.index, daywise.values)
plt.xticks(rotation=45)
plt.title("Crimes per Day of Week")
plt.show()

# 6. Crimes per Month
monthwise = df.groupby("Month").size()
plt.figure()
plt.plot(monthwise.index, monthwise.values, marker='o')
plt.xlabel("Month (1-12)")
plt.ylabel("Number of Crimes")
plt.title("Crimes per Month")
plt.show()

# 7. Top crime areas
if "Neighborhood" in df.columns:
    area = df.groupby("Neighborhood").size().sort_values(ascending=False)
    top10 = area.head(10)

    print("\n Top 10 Crime-prone Neighborhoods:")
    print(top10)

    plt.figure()
    plt.bar(top10.index, top10.values)
    plt.xticks(rotation=90)
    plt.title("Top 10 Crime Areas")
    plt.show()
else:
    print("\n 'Neighborhood' column not found. Skipping area analysis.")

# 8. Safest & Riskiest Hours
safest = hourly.sort_values().head(5)
riskiest = hourly.sort_values(ascending=False).head(5)

print("\n Safest Hours (least crime):")
print(safest)

print("\n Riskiest Hours (most crime):")
print(riskiest)

# 9. Machine Learning Model (Decision Tree)
def time_slot(h):
    if 6 <= h < 12:
        return "Morning"
    elif 12 <= h < 18:
        return "Afternoon"
    elif 18 <= h < 24:
        return "Night"
    else:
        return "Midnight"

df["TimeSlot"] = df["Hour"].apply(time_slot)

if "Neighborhood" in df.columns:
    le_area = LabelEncoder()
    le_slot = LabelEncoder()

    df["Neighborhood_enc"] = le_area.fit_transform(df["Neighborhood"].astype(str))
    df["TimeSlot_enc"] = le_slot.fit_transform(df["TimeSlot"].astype(str))

    # ML Inputs
    X = df[["Neighborhood_enc"]]
    y = df["TimeSlot_enc"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n ML Model Accuracy:", round(accuracy * 100, 2), "%")

    #  USER INPUT PREDICTION
    print("\n Enter a Neighborhood Name to Predict Crime Time")
    user_input = input("Neighborhood: ")

    if user_input in df["Neighborhood"].unique():
        encoded_value = le_area.transform([user_input])
        prediction = model.predict([[encoded_value[0]]])
        result = le_slot.inverse_transform(prediction)[0]
        print(f"\n Prediction: Crimes in '{user_input}' are most likely in the '{result}' time.")
    else:
        print("\n Neighborhood not found in dataset.")

else:
    print("\n ML skipped because 'Neighborhood' column not found.")

print("\n PROJECT COMPLETED SUCCESSFULLY!")
