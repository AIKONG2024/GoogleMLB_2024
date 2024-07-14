import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
import optuna
import logging

# 디렉토리 설정
dir = './health_insurance/data/'
output_dir = dir + "output/"
os.makedirs(output_dir, exist_ok=True)

# 로그 설정
log_filename = os.path.join(output_dir, 'model_optimization.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

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

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optuna를 사용한 하이퍼파라미터 최적화 함수 정의
def optimize_xgb(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'eta': trial.suggest_float('eta', 0.005, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }
    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model.fit(x_train, y_train)
    y_test_prob = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, y_test_prob)
    return auc

def optimize_catboost(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'random_strength': trial.suggest_int('random_strength', 0, 100),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0),
    }
    model = CatBoostClassifier(**param, verbose=0)
    model.fit(x_train, y_train)
    y_test_prob = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, y_test_prob)
    return auc

def optimize_lgb(trial):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'max_depth': trial.suggest_int('max_depth', -1, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = lgb.LGBMClassifier(**param)
    model.fit(x_train, y_train)
    y_test_prob = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, y_test_prob)
    return auc

# 데이터 분할
x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.3, shuffle=True, random_state=42, stratify=y_train)

# Optuna 스터디 생성 및 최적화
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(optimize_xgb, n_trials=1)
logging.info(f"XGBoost Best Parameters: {study_xgb.best_params}")
logging.info(f"XGBoost Best AUC: {study_xgb.best_value}")

study_catboost = optuna.create_study(direction='maximize')
study_catboost.optimize(optimize_catboost, n_trials=1)
logging.info(f"CatBoost Best Parameters: {study_catboost.best_params}")
logging.info(f"CatBoost Best AUC: {study_catboost.best_value}")

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(optimize_lgb, n_trials=1)
logging.info(f"LightGBM Best Parameters: {study_lgb.best_params}")
logging.info(f"LightGBM Best AUC: {study_lgb.best_value}")

# 최적 하이퍼파라미터로 모델 생성
xgb_best = xgb.XGBClassifier(**study_xgb.best_params, use_label_encoder=False)
catboost_best = CatBoostClassifier(**study_catboost.best_params, verbose=0)
lgb_best = lgb.LGBMClassifier(**study_lgb.best_params)

# 하드 보팅과 소프트 보팅을 사용하는 앙상블 모델 정의
voting_hard = VotingClassifier(estimators=[
    ('xgb', xgb_best),
    ('catboost', catboost_best),
    ('lgb', lgb_best)
], voting='hard')

voting_soft = VotingClassifier(estimators=[
    ('xgb', xgb_best),
    ('catboost', catboost_best),
    ('lgb', lgb_best)
], voting='soft')

# 앙상블 모델 학습 및 평가 함수 정의
def model_evaluation(model, x_train, y_train, x_test, y_test, model_name, is_hard_voting=False):
    model.fit(x_train, y_train)
    if is_hard_voting:
        y_test_pred = model.predict(x_test)
        auc_score = roc_auc_score(y_test, y_test_pred)
    else:
        y_test_prob = model.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_test_prob)
    logging.info(f"AUC Score of {model_name} model on Validation Data is: {auc_score:.4f}")
    return auc_score

# 앙상블 모델 평가
print("\nEvaluating Hard Voting Classifier:")
hard_auc = model_evaluation(voting_hard, x_train, y_train, x_val, y_val, "Hard Voting", is_hard_voting=True)

print("\nEvaluating Soft Voting Classifier:")
soft_auc = model_evaluation(voting_soft, x_train, y_train, x_val, y_val, "Soft Voting")

# 가장 높은 AUC 점수를 가진 앙상블 모델 선택
if hard_auc > soft_auc:
    best_ensemble = voting_hard
    best_ensemble_name = "Hard Voting"
else:
    best_ensemble = voting_soft
    best_ensemble_name = "Soft Voting"

logging.info(f"The best ensemble model is {best_ensemble_name} with AUC score of {max(hard_auc, soft_auc):.4f}")

# 최종 테스트 데이터 예측
best_ensemble.fit(X_train_scaled, y_train)
if best_ensemble_name == "Hard Voting":
    test_probs = best_ensemble.predict(X_test_scaled)
else:
    test_probs = best_ensemble.predict_proba(X_test_scaled)[:, 1]

# sample_submission.csv 로 결과 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = os.path.join(output_dir, f'sample_submission_{best_ensemble_name}_{max(hard_auc, soft_auc):.4f}_{timestamp}.csv')
df_submission = pd.read_csv(dir + 'sample_submission.csv')
df_submission['Response'] = test_probs
df_submission.to_csv(output_file_path, index=False)

logging.info(f"Predictions saved to {output_file_path}")
print(f"Predictions saved to {output_file_path}")
