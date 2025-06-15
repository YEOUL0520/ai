
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from catboost import CatBoostClassifier

df = pd.read_csv("./data/training_data.csv")
df["Ori_X_sin"] = np.sin(np.radians(df["Ori_X"]))
df["Ori_X_cos"] = np.cos(np.radians(df["Ori_X"]))
df["Position_Label"] = df["Position_X"].astype(str) + "_" + df["Position_Y"].astype(str)

le = LabelEncoder()
df["Label_encoded"] = le.fit_transform(df["Position_Label"])

X = df[["Mag_X", "Mag_Y", "Mag_Z", "Ori_X_sin", "Ori_X_cos"]]
y = df["Label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3,
    loss_function='MultiClass',
    auto_class_weights='Balanced',
    task_type='GPU',
    verbose=100,
    random_seed=42
)
model.fit(X_train, y_train)

bundle = {"model": model, "label_encoder": le}
joblib.dump(bundle, "model_and_encoder.pkl")
print("✅ 모델과 LabelEncoder 저장 완료: model_and_encoder.pkl")