import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df_banco = pd.read_csv("data/banco_dados.csv")
# Separating training and testing
X = df_banco[['AGE', 'ACCOUNT_BALANCE', 'HAS_CREDIT_CARD', 'ACTIVE', 'SALARY']]
y = df_banco['LEFT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
oversampler = RandomOverSampler(random_state=42)
X_train_scaled, y_train = oversampler.fit_resample(X_train_scaled, y_train)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Creating the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the Random Forest model
rf_model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluating the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

# Random Forest Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix:")
print(conf_matrix_rf)

# Random Forest Classification Report
class_report_rf = classification_report(y_test, y_pred_rf)
print("\nRandom Forest Classification Report:")
print(class_report_rf)
import joblib
import pickle

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f, protocol=2)
    
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f, protocol=2)
