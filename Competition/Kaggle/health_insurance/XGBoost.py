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

# 모델 학습 및 평가 함수 정의
def model_prediction(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)
    x_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)[:, 1]

    a = accuracy_score(y_train, x_train_pred) * 100
    b = accuracy_score(y_test, x_test_pred) * 100
    c = precision_score(y_test, x_test_pred)
    d = recall_score(y_test, x_test_pred)
    e = roc_auc_score(y_test, y_test_prob)
    
    print(f"Accuracy_Score of {model} model on Training Data is:", a)
    print(f"Accuracy_Score of {model} model on Testing Data is:", b)
    print(f"Precision Score of {model} model is:", c)
    print(f"Recall Score of {model} model is:", d)
    print(f"AUC Score of {model} model is:", e)
    
    # cm = confusion_matrix(y_test, x_test_pred)
    # plt.figure(figsize=(8,4))
    # sns.heatmap(cm, annot=True, fmt="g", cmap="Greens")
    # plt.show()

# 모델 학습 및 평가
x_train, x_val, y_train, y_val = train_test_split(X_train_scaled, y_train_smote, test_size=0.15, random_state=42, stratify=y_train_smote)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_prediction(model, x_train, y_train, x_val, y_val)

# 테스트 데이터 예측
test_prob = model.predict_proba(X_test_scaled)[:, 1]

# sample_submission.csv 로 결과 저장
df_submission = pd.read_csv(dir + 'sample_submission.csv')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = os.path.join(output_dir, f'sample_submission_{timestamp}.csv')
df_submission['Response'] = test_prob
df_submission.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")