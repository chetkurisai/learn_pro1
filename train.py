from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
import os

def train_and_save_model(X_train, y_train, model_output_path):

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_output_path, 'model.joblib'))
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-output-path', dest='model_output_path',
                        default=r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\model', type=str, help='GCS location to write model artifacts')
    args = parser.parse_args()

    import numpy as np
    X_train = np.load(r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\X_train.npy')
    y_train = np.load(r'D:\ml_prjects\Projects\Titanic_streamlit_deploy\y_train.npy')
    train_and_save_model(X_train, y_train, args.model_output_path)