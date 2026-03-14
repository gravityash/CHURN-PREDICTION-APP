import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv('customer_churn_data.csv')

# Fill NA for categorical
df['InternetService'] = df['InternetService'].fillna('')

# Drop duplicates
df = df.drop_duplicates()

# Encode categorical columns
df['Gender'] = df['Gender'].apply(lambda x: 1 if x.lower() == 'female' else 0)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

# Prepare features and target
features = ['Age', 'Gender', 'Tenure', 'MonthlyCharges']
X = df[features]
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

def evaluate_model(model):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')
    return model

# Train models and check accuracy
log_model = evaluate_model(LogisticRegression())
knn_model = evaluate_model(KNeighborsClassifier(n_neighbors=7))
svc_model = evaluate_model(SVC(C=0.01, kernel='linear'))
dt_model = evaluate_model(DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2))
rf_model = evaluate_model(RandomForestClassifier(n_estimators=32, max_features=2, bootstrap=True))

# Save the best model (e.g., SVC)
joblib.dump(svc_model, 'model.pkl')

print("Feature order:", features)
