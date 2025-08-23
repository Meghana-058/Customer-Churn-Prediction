import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json

# Load raw dataset
df = pd.read_csv('../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean TotalCharges column
df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Encode categorical columns except customerID
for col in df.select_dtypes(include='object').columns:
    if col != 'customerID':
        df[col] = LabelEncoder().fit_transform(df[col])

# Create features and target
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']

# Train initial model for feature importance
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get top 5 features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feat_importances.nlargest(5).index.tolist()
print("Top features:", top_features)

# Retrain model with top features
X_top = X[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)
model_top = RandomForestClassifier(random_state=42)
model_top.fit(X_train, y_train)

# Save model and feature list
joblib.dump(model_top, '../Data/churn_model_top5.pkl')
with open('../Data/top_features.json', 'w') as f:
    json.dump(top_features, f)
