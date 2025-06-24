import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from matplotlib import pyplot as plt

# MAP@3 계산
def mean_average_precision(y_true, y_probs, top_k=3):
    ap_scores = []
    for true_label, prob in zip(y_true, y_probs):
        top_k_preds = np.argsort(prob)[-top_k:][::-1]
        if true_label in top_k_preds:
            rank = np.where(top_k_preds == true_label)[0][0] + 1
            ap_scores.append(1 / rank)
        else:
            ap_scores.append(0)
    return np.mean(ap_scores)

# 데이터 df에 로딩 및 라벨 생성
df = pd.read_csv("training_data.csv")
df["Position_Label"] = df["filename"].astype(str)

# 라벨 인코딩
le = LabelEncoder()
df["Label_encoded"] = le.fit_transform(df["Position_Label"])

# 입력 피처 설정
feature_cols = ["Mag_X", "Mag_Y", "Mag_Z", "Ori_X"]
X = df[feature_cols]
y = df["Label_encoded"]

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할 (70/20/10, stratified)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.10, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=2/9, random_state=42, stratify=y_temp
)

# 모델 초기화 및 학습
model = CatBoostClassifier(
    iterations=10000,
    depth = 8,
    early_stopping_rounds=10,
    learning_rate=0.0003, # 일단 이걸로 테스트
    loss_function='MultiClass',
    auto_class_weights='Balanced',
    l2_leaf_reg=3,
    random_seed=42,
    verbose=10,
    task_type="GPU"
)
print("\n---모델 학습 시작---")

# 학습 + 손실 기록
model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=10)
results = model.get_evals_result()

# 손실 곡선 저장해 손실율 확인 (제대로 학습 진행되는지 확인 위한 용도)
plt.figure(figsize=(10, 4))
plt.plot(results['learn']['MultiClass'], label="Train MultiClass Loss")
plt.plot(results['validation']['MultiClass'], label="Validation MultiClass Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("CatBoost 학습 손실 곡선 (MultiClass)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve5.png")
plt.close()
print("---그래프 저장: loss_curve4.png---")

# 학습시킨 모델 저장
bundle = {
    "model": model,
    "label_encoder": le,
    "scaler": scaler,
    "feature_columns": feature_cols
}
joblib.dump(bundle, "catboost_model_and_encoder_v4.pkl")

# 예측
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
y_pred_label = le.inverse_transform(y_pred.astype(int))
y_true_label = le.inverse_transform(y_test)

# 결과 DataFrame
test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df["True_Label"] = y_true_label
test_df["Pred_Label"] = y_pred_label
test_df["Top1_Correct"] = test_df["True_Label"] == test_df["Pred_Label"]

# Top-3 포함 여부
top3_correct = []
for true_idx, proba in zip(y_test, y_proba):
    top3 = np.argsort(proba)[-3:][::-1]
    top3_correct.append(true_idx in top3)
test_df["Top3_Correct"] = top3_correct

# 분류
df_top1 = test_df[test_df["Top1_Correct"]]
df_top3_only = test_df[(test_df["Top3_Correct"]) & (~test_df["Top1_Correct"])]
df_wrong = test_df[~test_df["Top3_Correct"]]

# 성능 지표 출력
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
map_score = mean_average_precision(y_test, y_proba, top_k=3)

print("\n---예측 성능 요약---")
print("=" * 50)
print(f"Top-1 정확도: {test_df['Top1_Correct'].mean():.2%} ({len(df_top1)} / {len(test_df)})")
print(f"Top-3 정확도: {test_df['Top3_Correct'].mean():.2%} ({len(df_top1)+len(df_top3_only)} / {len(test_df)})")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"MAP@3: {map_score:.4f}")

# 대표 샘플 출력
print("\n정확히 맞춘 샘플 (Top-1)")
print(df_top1[['True_Label', 'Pred_Label']].head(10))

print("\nTop-3 안에 있지만 Top-1은 틀린 샘플")
print(df_top3_only[['True_Label', 'Pred_Label']].head(10))

print("\nTop-3에도 못 들어간 샘플")
print(df_wrong[['True_Label', 'Pred_Label']].head(10))
