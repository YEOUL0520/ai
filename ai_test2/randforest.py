import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ 데이터 로딩 및 라벨 생성
df = pd.read_csv("training_data.csv")
df["Position_Label"] = df["filename"].astype(str)

# ✅ 라벨 인코딩
le = LabelEncoder()
df["Label_encoded"] = le.fit_transform(df["Position_Label"])

# ✅ 입력 피처
feature_cols = ["Mag_X", "Mag_Y", "Mag_Z", "Ori_X"]
X = df[feature_cols]
y = df["Label_encoded"]

# ✅ 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ 랜덤포레스트 초기화
model = RandomForestClassifier(
    n_estimators=0,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# ✅ 트리 반복적으로 학습

total_trees = 200
step = 10

print("\n🌲 모델 학습 시작")

for i in range(0, total_trees, step):
    model.set_params(n_estimators=i + step)
    model.fit(X_train, y_train)
    print(f"✅ {i + step}개 트리 학습 완료")

# ✅ 모델 저장
bundle = {
    "model": model,
    "label_encoder": le,
    "scaler": scaler,
    "feature_columns": feature_cols
}
joblib.dump(bundle, "randforest_model_and_encoder_v1.pkl")

# ✅ 예측 수행
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_pred_label = le.inverse_transform(y_pred)
y_true_label = le.inverse_transform(y_test)

# ✅ 결과 저장
test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df["True_Label"] = y_true_label
test_df["Pred_Label"] = y_pred_label
test_df["Top1_Correct"] = test_df["True_Label"] == test_df["Pred_Label"]

# ✅ Top-3 정답 포함 여부
top3_correct = []
for true_idx, proba in zip(y_test, y_proba):
    top_k = min(len(proba), 3)
    top3 = np.argsort(proba)[-top_k:][::-1]
    top3_correct.append(true_idx in top3)
test_df["Top3_Correct"] = top3_correct

# ✅ 결과 분류
df_top1 = test_df[test_df["Top1_Correct"]]
df_top3_only = test_df[(test_df["Top3_Correct"]) & (~test_df["Top1_Correct"])]
df_wrong = test_df[~test_df["Top3_Correct"]]

# ✅ 전체 성능 요약 출력
print("\n📊 예측 성능 요약")
print("=" * 40)
print(f"🎯 Top-1 정확도: {test_df['Top1_Correct'].mean():.2%} ({len(df_top1)} / {len(test_df)})")
print(f"🔁 Top-3 정확도: {test_df['Top3_Correct'].mean():.2%} ({len(test_df[test_df['Top3_Correct']])} / {len(test_df)})")

# ✅ 정답 그룹별 출력
print("\n🎯 정확히 맞춘 샘플 (Top-1)")
print(df_top1[['True_Label', 'Pred_Label']].head(10))

print("\n🔄 Top-3 안에 있지만 Top-1은 틀린 샘플")
print(df_top3_only[['True_Label', 'Pred_Label']].head(10))

print("\n❌ Top-3에도 못 들어간 샘플")
print(df_wrong[['True_Label', 'Pred_Label']].head(10))

# ✅ 위치별 상세 오인식 분석
detailed_summary = []

for label in test_df["True_Label"].unique():
    subset = test_df[test_df["True_Label"] == label]
    count = len(subset)
    top1_acc = subset["Top1_Correct"].mean()
    top3_acc = subset["Top3_Correct"].mean()

    wrong_subset = subset[~subset["Top1_Correct"]]
    top_wrong_preds = (
        wrong_subset["Pred_Label"].value_counts()
        .head(3)
        .to_dict()
    )

    detailed_summary.append({
        "True_Label": label,
        "Sample_Count": count,
        "Top1_Accuracy": top1_acc,
        "Top3_Accuracy": top3_acc,
        "Top_Misclass_1": list(top_wrong_preds.keys())[0] if len(top_wrong_preds) > 0 else None,
        "Top_Misclass_1_Count": list(top_wrong_preds.values())[0] if len(top_wrong_preds) > 0 else 0,
        "Top_Misclass_2": list(top_wrong_preds.keys())[1] if len(top_wrong_preds) > 1 else None,
        "Top_Misclass_2_Count": list(top_wrong_preds.values())[1] if len(top_wrong_preds) > 1 else 0,
        "Top_Misclass_3": list(top_wrong_preds.keys())[2] if len(top_wrong_preds) > 2 else None,
        "Top_Misclass_3_Count": list(top_wrong_preds.values())[2] if len(top_wrong_preds) > 2 else 0,
    })

# ✅ 출력
detailed_df = pd.DataFrame(detailed_summary).sort_values(by="True_Label")

print("\n📌 위치별 오인식 상세 분석")
print("=" * 50)
for _, row in detailed_df.iterrows():
    print(f"📍 위치: {row['True_Label']}")
    print(f"   - 샘플 수: {row['Sample_Count']}")
    print(f"   - Top-1 정확도: {row['Top1_Accuracy']:.2%}")
    print(f"   - Top-3 정확도: {row['Top3_Accuracy']:.2%}")
    print(f"   - ❌ 자주 틀리는 위치:")
    if row["Top_Misclass_1"]:
        print(f"       • {row['Top_Misclass_1']} ({row['Top_Misclass_1_Count']}회)")
    if row["Top_Misclass_2"]:
        print(f"       • {row['Top_Misclass_2']} ({row['Top_Misclass_2_Count']}회)")
    if row["Top_Misclass_3"]:
        print(f"       • {row['Top_Misclass_3']} ({row['Top_Misclass_3_Count']}회)")
    print()