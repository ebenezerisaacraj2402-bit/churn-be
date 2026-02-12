import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/churn.csv")

# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Fill missing values only for numeric columns
data.fillna(data.select_dtypes(include=['number']).mean(), inplace=True)

# Encode categorical columns
le_gender = LabelEncoder()
le_churn = LabelEncoder()

data['gender'] = le_gender.fit_transform(data['gender'])
data['Churn'] = le_churn.fit_transform(data['Churn'])

# Features & Target
X = data[['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
y = data['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
