import pandas as pd

# CSV 파일 불러오기
training_df = pd.read_csv('training_data.csv')
test_df = pd.read_csv('test_data.csv')

results = []

for _, test_row in test_df.iterrows():
    pos_x = test_row['Position_X']
    pos_y = test_row['Position_Y']
    ori_x = test_row['Ori_X']

    candidates = training_df[
        (training_df['Position_X'] == pos_x) &
        (training_df['Position_Y'] == pos_y)
    ]

    if candidates.empty:
        continue

    candidates = candidates.copy()
    candidates['Ori_X_diff'] = (candidates['Ori_X'] - ori_x).abs()
    best_match = candidates.loc[candidates['Ori_X_diff'].idxmin()]

    mag_x_diff = abs(test_row['Mag_X'] - best_match['Mag_X'])
    mag_y_diff = abs(test_row['Mag_Y'] - best_match['Mag_Y'])
    mag_z_diff = abs(test_row['Mag_Z'] - best_match['Mag_Z'])

    results.append({
        'Position_X': pos_x,
        'Position_Y': pos_y,
        'Test_Ori_X': ori_x,
        'Train_Ori_X': best_match['Ori_X'],
        'Diff_Ori_X': best_match['Ori_X_diff'],
        'Mag_X_test': test_row['Mag_X'],
        'Mag_X_train': best_match['Mag_X'],
        'Mag_X_diff': mag_x_diff,
        'Mag_Y_test': test_row['Mag_Y'],
        'Mag_Y_train': best_match['Mag_Y'],
        'Mag_Y_diff': mag_y_diff,
        'Mag_Z_test': test_row['Mag_Z'],
        'Mag_Z_train': best_match['Mag_Z'],
        'Mag_Z_diff': mag_z_diff,
        'Total_Mag_diff': mag_x_diff + mag_y_diff + mag_z_diff
    })

# 결과 DataFrame 생성
result_df = pd.DataFrame(results)

# Mag 차이가 큰 상위 100개 추출
top_100_diff = result_df.sort_values(by='Total_Mag_diff', ascending=False).head(100)
print("\n=== Mag 차이 큰 상위 100개 ===")
print(top_100_diff)

# 전체 평균 계산
avg_mag_x_diff = result_df['Mag_X_diff'].mean()
avg_mag_y_diff = result_df['Mag_Y_diff'].mean()
avg_mag_z_diff = result_df['Mag_Z_diff'].mean()
avg_total_mag_diff = result_df['Total_Mag_diff'].mean()

print("\n=== 전체 평균 차이 ===")
print(f"Mag_X 평균 차이: {avg_mag_x_diff:.4f}")
print(f"Mag_Y 평균 차이: {avg_mag_y_diff:.4f}")
print(f"Mag_Z 평균 차이: {avg_mag_z_diff:.4f}")
print(f"총합 기준 평균 차이: {avg_total_mag_diff:.4f}")
