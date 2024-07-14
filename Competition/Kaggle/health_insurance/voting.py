import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# 디렉토리 설정
dir = './health_insurance/data/'
output_dir = dir + "output/"
os.makedirs(output_dir, exist_ok=True)

# 데이터 로드
df_train = pd.read_csv(dir + 'train_s.csv')
df_test = pd.read_csv(dir + 'test.csv')

# 전처리 함수 정의
def preprocess_data(df):
    def veh_a(Vehicle_Damage):
        return 1 if Vehicle_Damage == 'Yes' else 0

    if 'Vehicle_Damage' in df.columns:
        df['Vehicle_Damages'] = df['Vehicle_Damage'].apply(veh_a)
        df.drop(['Vehicle_Damage'], axis=1, inplace=True)
    
    df['Vehicle_Age'] = df['Vehicle_Age'].astype('category')
    df = pd.get_dummies(df, columns=['Vehicle_Age', 'Gender'], drop_first=True)
    return df

# 전처리 적용
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

# 특징 및 레이블 분리
X_train = df_train.drop(columns=['Response', 'id'])
y_train = df_train['Response']
X_test = df_test.drop(columns=['id'])

# SMOTE 적용
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# 모델 정의
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
catboost_model = CatBoostClassifier(verbose=0)
lgb_model = lgb.LGBMClassifier()

# 모델 학습 및 평가 함수 정의
def model_evaluation(model, x_train, y_train, x_test, y_test, model_name):
    model.fit(x_train, y_train)
    y_test_prob = model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_test_prob)
    print(f"AUC Score of {model_name} model on Validation Data is: {auc_score:.4f}")
    return auc_score, y_test_prob

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, y_train_smote, test_size=0.15, random_state=42)

# 개별 모델 평가
models = {
    "XGBoost": xgb_model,
    "CatBoost": catboost_model,
    "LightGBM": lgb_model
}

best_model_name = None
best_model = None
best_score = 0
best_probs = None

for model_name, model in models.items():
    auc_score, probs = model_evaluation(model, x_train, y_train, x_val, y_val, model_name)
    if auc_score > best_score:
        best_score = auc_score
        best_model_name = model_name
        best_model = model
        best_probs = probs

print(f"The best individual model is {best_model_name} with AUC score of {best_score:.4f}")

# 하드 보팅과 소프트 보팅을 사용하는 앙상블 모델 정의
voting_hard = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('catboost', catboost_model),
    ('lgb', lgb_model)
], voting='hard')

voting_soft = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('catboost', catboost_model),
    ('lgb', lgb_model)
], voting='soft')

# 앙상블 모델 학습 및 평가
print("\nEvaluating Hard Voting Classifier:")
hard_auc, _ = model_evaluation(voting_hard, x_train, y_train, x_val, y_val, "Hard Voting")

print("\nEvaluating Soft Voting Classifier:")
soft_auc, _ = model_evaluation(voting_soft, x_train, y_train, x_val, y_val, "Soft Voting")

# 가장 높은 AUC 점수를 가진 앙상블 모델 선택
if hard_auc > soft_auc:
    best_ensemble = voting_hard
    print(f"\nThe best ensemble model is Hard Voting with AUC score of {hard_auc:.4f}")
else:
    best_ensemble = voting_soft
    print(f"\nThe best ensemble model is Soft Voting with AUC score of {soft_auc:.4f}")

# 최종 테스트 데이터 예측
best_ensemble.fit(X_train_scaled, y_train_smote)
test_probs = best_ensemble.predict_proba(X_test_scaled)[:, 1]

# sample_submission.csv 로 결과 저장
df_submission = pd.read_csv(dir + 'sample_submission.csv')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = os.path.join(output_dir, f'sample_submission_{timestamp}.csv')
df_submission['Response'] = test_probs
df_submission.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
