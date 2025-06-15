import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ✅ 데이터 로딩 및 라벨 생성
df = pd.read_csv("./data/training_data.csv")
df["Position_Label"] = df["Position_X"].astype(str) + "_" + df["Position_Y"].astype(str)

# ✅ 라벨 인코딩
le = LabelEncoder()
df["Label_encoded"] = le.fit_transform(df["Position_Label"])

# ✅ 입력 피처 및 정규화
X = df[["Mag_X", "Mag_Y", "Mag_Z", "Ori_X"]]
y = df["Label_encoded"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ XGBoost 모델 정의 및 GPU 설정 포함
model = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.0001,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="gpu_hist",        # ✅ GPU 사용
    predictor="gpu_predictor",     # ✅ GPU 사용
    verbosity=1,
    random_state=42
)

print("\n🚀 XGBoost (GPU) 모델 학습 시작")
model.fit(X_train, y_train)

# ✅ 모델 저장
bundle = {
    "model": model,
    "label_encoder": le,
    "scaler": scaler,
    "feature_columns": ["Mag_X", "Mag_Y", "Mag_Z", "Ori_X"]
}
joblib.dump(bundle, "model_and_encoder.pkl")

# ✅ 성능 출력
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\n📊 학습 완료")
print("=" * 40)
print("✅ 학습 정확도: {:.2%}".format(train_acc))
print("✅ 검증 정확도: {:.2%}".format(test_acc))
print("✅ 모델과 인코더 저장 완료: model_and_encoder.pkl")
