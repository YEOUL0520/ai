import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ✅ 1. 테스트 데이터 로딩
df = pd.read_csv("./data/test_data.csv")

# ✅ 2. 라벨 및 파생변수 생성 (학습과 동일하게)
df["Position_Label"] = df["Position_X"].astype(str) + "_" + df["Position_Y"].astype(str)
df["Ori_X_sin"] = np.sin(np.radians(df["Ori_X"]))
df["Ori_X_cos"] = np.cos(np.radians(df["Ori_X"]))

# ✅ 3. 모델 및 인코더 로드
bundle = joblib.load("model_and_encoder.pkl")
model = bundle["model"]
le = bundle["label_encoder"]

# ✅ 4. 라벨 인코딩
df["Label_encoded"] = le.transform(df["Position_Label"])

# ✅ 5. 피처 선택 (학습과 동일한 순서로)
X = df[["Mag_X", "Mag_Y", "Mag_Z", "Ori_X_sin", "Ori_X_cos"]]
y_true = df["Label_encoded"]

# ✅ 6. 예측 수행
y_pred = model.predict(X).ravel()
y_proba = model.predict_proba(X)

# ✅ 7. 복호화 및 위치 오차 계산
y_pred_label = le.inverse_transform(y_pred.astype(int))
y_true_label = le.inverse_transform(y_true.astype(int))

def coord_from_label(label):
    x, y = map(int, label.split("_"))
    return np.array([x, y])

errors = [
    np.linalg.norm(coord_from_label(p) - coord_from_label(t))
    for p, t in zip(y_pred_label, y_true_label)
]

# ✅ 8. 결과 출력
print("\n📊 예측 성능 요약")
print("=" * 40)
print("🎯 Top-1 정확도: {:.2%}".format(accuracy_score(y_true, y_pred)))
print("🔁 Top-3 정확도: {:.2%}".format(
    top_k_accuracy_score(y_true, y_proba, k=3, labels=np.arange(len(le.classes_)))
))
print("📏 평균 위치 오차: {:.2f} 셀".format(np.mean(errors)))
