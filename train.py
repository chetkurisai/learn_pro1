from sklearn.ensemble import RandomForestRegressor
import joblib
import argparse
import os
import numpy as np

def train_and_save_model(X_train, y_train, model_output_path):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_output_path, 'model.joblib'))
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-output-path', dest='model_output_path',
                        default=r'D:\ml_prjects\Projects\Ml_ops_Proj1\model', type=str, help='Path to save model')
    args = parser.parse_args()

    X_train = np.load(r'D:\ml_prjects\Projects\Ml_ops_Proj1\X_train.npy')
    y_train = np.load(r'D:\ml_prjects\Projects\Ml_ops_Proj1\y_train.npy')
    train_and_save_model(X_train, y_train, args.model_output_path)
