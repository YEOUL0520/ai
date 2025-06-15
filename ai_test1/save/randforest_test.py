import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ✅ 데이터 로딩 및 라벨 생성
df = pd.read_csv("./data/test_data.csv")
df["Position_Label"] = df["Position_X"].astype(str) + "_" + df["Position_Y"].astype(str)

# ✅ 모델 및 전처리기 불러오기
bundle = joblib.load("model_and_encoder.pkl")
model = bundle["model"]
le = bundle["label_encoder"]
scaler = bundle["scaler"]
feature_cols = bundle["feature_columns"]

# ✅ 라벨 인코딩
df["Label_encoded"] = le.transform(df["Position_Label"])

# ✅ 입력 정규화
X = scaler.transform(df[feature_cols])
y_true = df["Label_encoded"]

# ✅ 예측 수행
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
y_pred_label = le.inverse_transform(y_pred)
y_true_label = le.inverse_transform(y_true)

# ✅ 위치 오차 계산
def coord_from_label(label):
    x, y = map(int, label.split("_"))
    return np.array([x, y])

errors = [
    np.linalg.norm(coord_from_label(p) - coord_from_label(t))
    for p, t in zip(y_pred_label, y_true_label)
]

# ✅ 결과 저장
df['Pred_Label'] = y_pred_label
df['True_Label'] = y_true_label
df['Top1_Correct'] = df['Pred_Label'] == df['True_Label']
df['Position_Error'] = errors

# ✅ Top-3 정답 포함 여부
top3_correct = []
for true_idx, proba in zip(y_true, y_proba):
    top3 = np.argsort(proba)[-3:][::-1]
    top3_correct.append(true_idx in top3)
df['Top3_Correct'] = top3_correct

# ✅ 결과 분류
df_top1 = df[df['Top1_Correct']]
df_top3_only = df[(df['Top3_Correct']) & (~df['Top1_Correct'])]
df_wrong = df[~df['Top3_Correct']]

# ✅ 전체 성능 요약 출력
print("\n📊 예측 성능 요약")
print("=" * 40)
print(f"🎯 Top-1 정확도: {df['Top1_Correct'].mean():.2%} ({len(df_top1)} / {len(df)})")
print(f"🔁 Top-3 정확도: {df['Top3_Correct'].mean():.2%} ({len(df[df['Top3_Correct']])} / {len(df)})")
print(f"📏 평균 위치 오차: {df['Position_Error'].mean():.2f} 셀")

# ✅ 정답 그룹별 출력
print("\n🎯 정확히 맞춘 샘플 (Top-1)")
print(df_top1[['True_Label', 'Pred_Label', 'Position_Error']].head(10))

print("\n🔄 Top-3 안에 있지만 Top-1은 틀린 샘플")
print(df_top3_only[['True_Label', 'Pred_Label', 'Position_Error']].head(10))

print("\n❌ Top-3에도 못 들어간 샘플")
print(df_wrong[['True_Label', 'Pred_Label', 'Position_Error']].head(10))

# ✅ 위치별 상세 오인식 분석
detailed_summary = []

for label in df['True_Label'].unique():
    subset = df[df['True_Label'] == label]
    count = len(subset)
    top1_acc = subset['Top1_Correct'].mean()
    top3_acc = subset['Top3_Correct'].mean()
    avg_error = subset['Position_Error'].mean()

    # 오답만 추출해서 어디로 많이 잘못 갔는지 확인
    wrong_subset = subset[~subset['Top1_Correct']]
    top_wrong_preds = (
        wrong_subset['Pred_Label'].value_counts()
        .head(3)
        .to_dict()
    )

    detailed_summary.append({
        "True_Label": label,
        "Sample_Count": count,
        "Top1_Accuracy": top1_acc,
        "Top3_Accuracy": top3_acc,
        "Avg_Position_Error": avg_error,
        "Top_Misclass_1": list(top_wrong_preds.keys())[0] if len(top_wrong_preds) > 0 else None,
        "Top_Misclass_1_Count": list(top_wrong_preds.values())[0] if len(top_wrong_preds) > 0 else 0,
        "Top_Misclass_2": list(top_wrong_preds.keys())[1] if len(top_wrong_preds) > 1 else None,
        "Top_Misclass_2_Count": list(top_wrong_preds.values())[1] if len(top_wrong_preds) > 1 else 0,
        "Top_Misclass_3": list(top_wrong_preds.keys())[2] if len(top_wrong_preds) > 2 else None,
        "Top_Misclass_3_Count": list(top_wrong_preds.values())[2] if len(top_wrong_preds) > 2 else 0
    })

# ✅ DataFrame 정리 및 출력
detailed_df = pd.DataFrame(detailed_summary).sort_values(by="True_Label")

# ✅ 출력
print("\n📌 위치별 오인식 상세 분석")
print("=" * 50)
for _, row in detailed_df.iterrows():
    print(f"📍 위치: {row['True_Label']}")
    print(f"   - 샘플 수: {row['Sample_Count']}")
    print(f"   - Top-1 정확도: {row['Top1_Accuracy']:.2%}")
    print(f"   - Top-3 정확도: {row['Top3_Accuracy']:.2%}")
    print(f"   - 평균 위치 오차: {row['Avg_Position_Error']:.2f} 셀")
    print(f"   - ❌ 자주 틀리는 위치:")
    if row['Top_Misclass_1']:
        print(f"       • {row['Top_Misclass_1']} ({row['Top_Misclass_1_Count']}회)")
    if row['Top_Misclass_2']:
        print(f"       • {row['Top_Misclass_2']} ({row['Top_Misclass_2_Count']}회)")
    if row['Top_Misclass_3']:
        print(f"       • {row['Top_Misclass_3']} ({row['Top_Misclass_3_Count']}회)")
    print()

