import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

# Load data correctly
df_path = r"D:\ml_prjects\Data Sets\sales_data.csv"  # Use raw string or double backslashes
df = pd.read_csv(df_path)  # Convert string to DataFrame

# Preprocessing (optional: Handle missing values, categorical encoding, etc.)
# df.fillna(df.mean(numeric_only=True), inplace=True)  
# df['sex'] = df['sex'].map({'male': 0, 'female': 1})
# df.drop(['embarked'], axis=1, inplace=True)

# Define Features (X) and Target (y)
X = df.drop('Units_Sold', axis=1)
y = df['Units_Sold']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column Transformer for preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor
joblib.dump(preprocessor, r'D:\ml_prjects\Projects\Ml_ops_Proj1\preprocessor.joblib')

# Save processed data
np.save(r'D:\ml_prjects\Projects\Ml_ops_Proj1\X_train.npy', X_train_processed)
np.save(r'D:\ml_prjects\Projects\Ml_ops_Proj1\y_train.npy', y_train.values)
np.save(r'D:\ml_prjects\Projects\Ml_ops_Proj1\X_test.npy', X_test_processed)
np.save(r'D:\ml_prjects\Projects\Ml_ops_Proj1\y_test.npy', y_test.values)

print("Data preprocessing complete.")
