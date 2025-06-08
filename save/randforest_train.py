import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 로딩 및 라벨 생성
df = pd.read_csv("./data/training_data.csv")
df["Position_Label"] = df["Position_X"].astype(str) + "_" + df["Position_Y"].astype(str)

# 라벨 인코딩
le = LabelEncoder()
df["Label_encoded"] = le.fit_transform(df["Position_Label"])

# 입력 피처
X = df[["Mag_X", "Mag_Y", "Mag_Z", "Ori_X"]]
y = df["Label_encoded"]

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 랜덤포레스트 초기화 (warm_start=True로 설정)
model = RandomForestClassifier(
    n_estimators=0,       # 처음은 0개로 시작
    max_depth=10,
    warm_start=True,      # 트리를 점진적으로 추가할 수 있도록 함
    random_state=42
)

# 총 트리 수
total_trees = 300
step = 10

print("\n🌲 모델 학습 시작")

for i in range(0, total_trees, step):
    model.set_params(n_estimators=i + step)  # 트리를 10개씩 추가
    model.fit(X_train, y_train)
    print(f"✅ {i + step}개 트리 학습 완료")

# 모델 저장
bundle = {
    "model": model,
    "label_encoder": le,
    "scaler": scaler,
    "feature_columns": ["Mag_X", "Mag_Y", "Mag_Z", "Ori_X"]
}
joblib.dump(bundle, "model_and_encoder.pkl")

# 성능 출력
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\n📊 학습 완료")
print("=" * 40)
print("✅ 학습 정확도: {:.2%}".format(train_acc))
print("✅ 검증 정확도: {:.2%}".format(test_acc))
print("✅ 모델과 인코더 저장 완료: model_and_encoder.pkl")
