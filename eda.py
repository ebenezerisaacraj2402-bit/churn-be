import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/churn.csv")

# Convert TotalCharges to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

print("Dataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# ------------------------------
# 1. Churn Distribution
# ------------------------------
plt.figure()
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

# ------------------------------
# 2. Monthly Charges vs Churn
# ------------------------------
plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=data)
plt.title("Monthly Charges vs Churn")
plt.savefig("monthly_vs_churn.png")
plt.show()

# ------------------------------
# 3. Tenure Distribution
# ------------------------------
plt.figure()
sns.histplot(data['tenure'], bins=20)
plt.title("Tenure Distribution")
plt.savefig("tenure_distribution.png")
plt.show()

# ------------------------------
# 4. Correlation Heatmap
# ------------------------------
plt.figure()
numeric_data = data.select_dtypes(include=['number'])
sns.heatmap(numeric_data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

print("EDA completed. Charts saved as PNG files.")
