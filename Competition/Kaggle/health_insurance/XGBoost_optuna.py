import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from datetime import datetime
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

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

# Optuna를 사용한 하이퍼파라미터 최적화
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }

    x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, y_train_smote, test_size=0.3, random_state=42)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)

    model = xgb.train(param, dtrain, evals=[(dval, 'val')], early_stopping_rounds=100, verbose_eval=False)
    preds = model.predict(dval)
    auc = roc_auc_score(y_val, preds)
    return auc

study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터로 모델 학습
best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['eval_metric'] = 'auc'

dtrain = xgb.DMatrix(X_train_scaled, label=y_train_smote)
final_model = xgb.train(best_params, dtrain, num_boost_round=1000)

# 테스트 데이터 예측
dtest = xgb.DMatrix(X_test_scaled)
test_predictions = final_model.predict(dtest)
test_predictions_binary = (test_predictions > 0.5).astype(int)

# 새로운 폴더 생성 및 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = os.path.join(output_dir, f'sample_submission_{timestamp}.csv')

df_submission = pd.read_csv(dir + 'sample_submission.csv')
df_submission['Response'] = test_predictions_binary
df_submission.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")

# Optuna 튜닝 결과 시각화 및 출력
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()
optuna.visualization.matplotlib.plot_parallel_coordinate(study)
plt.show()

print("Best parameters:", best_params)
print("Best score:", study.best_value)