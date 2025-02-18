import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import seaborn as sns

# Load data
df_path = r"D:\ml_prjects\Data Sets\titanic.csv"
df = pd.read_csv(df_path)

# Preprocessing
df.fillna(df.mean(numeric_only=True), inplace=True)  # Simple imputation
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df.drop(['Embarked'], axis=1, inplace=True)
df.drop(['PassengerId'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

# Feature engineering (example)
X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Column Transformer for preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor
#file formats to store the data: pickle (pkl), joblib, Orc, parquet
joblib.dump(preprocessor, r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\preprocessor.joblib')

# Save processed data (optional, for local training)
import numpy as np
np.save(r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\X_train.npy', X_train_processed)
np.save(r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\y_train.npy', y_train.values)
np.save(r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\X_test.npy', X_test_processed)
np.save(r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\y_test.npy', y_test.values)

print("Data preprocessing complete.")
