import numpy as np
import pandas as pd
import polars as pl
import os
import torch
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from datetime import datetime

# 설정
N_THREADS = os.cpu_count()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
np.random.seed(42)
torch.set_num_threads(N_THREADS)
if device == 'cuda':
    torch.cuda.set_device(0)

# 데이터 로드 및 전처리
orig_train = pl.read_csv('./health_insurance/data/train.csv')
train_n = pl.read_csv('./health_insurance/data/train_n.csv')
test = pl.read_csv('./health_insurance/data/test.csv')

train = pl.concat([train_n, orig_train])
target = 'Response'
test = test.with_columns(pl.lit(0).cast(pl.Int64).alias(target))

df = pl.concat([train, test])

df = df.with_columns([
    pl.col('Gender').replace({'Male': 0, 'Female': 1}).cast(pl.Int32),
    pl.col('Region_Code').cast(int),
    pl.col('Vehicle_Age').replace({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}).cast(pl.Int32),
    pl.col('Vehicle_Damage').replace({'No': 0, 'Yes': 1}).cast(pl.Int32),
    pl.col('Annual_Premium').cast(int),
    pl.col('Policy_Sales_Channel').cast(int)
])

df = df.with_columns([
    (pl.Series(pd.factorize((df['Previously_Insured'].cast(str) + df['Annual_Premium'].cast(str)).to_numpy())[0])).alias('Previously_Insured_Annual_Premium'),
    (pl.Series(pd.factorize((df['Previously_Insured'].cast(str) + df['Vehicle_Age'].cast(str)).to_numpy())[0])).alias('Previously_Insured_Vehicle_Age'),
    (pl.Series(pd.factorize((df['Previously_Insured'].cast(str) + df['Vehicle_Damage'].cast(str)).to_numpy())[0])).alias('Previously_Insured_Vehicle_Damage'),
    (pl.Series(pd.factorize((df['Previously_Insured'].cast(str) + df['Vintage'].cast(str)).to_numpy())[0])).alias('Previously_Insured_Vintage')
])

train = df[:train.shape[0]].to_pandas()
test = df[train.shape[0]:].to_pandas()

X = train.drop(columns=['id', target])
y = train[target]
X_test = test.drop(columns=['id', target])

# 특정 fold에만 데이터 추가
def add_data_to_fold(X, y, fold, train_n):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if i == fold:
            train_n_df = train_n.to_pandas()  # train_n을 pandas DataFrame으로 변환
            X_train = pd.concat([X_train, train_n_df.drop(['Response'], axis=1)], axis=0)
            y_train = pd.concat([y_train, train_n_df['Response']], axis=0)
        
        yield X_train, y_train, X_val, y_val

# TabularAutoML 설정 및 학습
fold = 0  # 특정 fold를 지정
for X_train, y_train, X_val, y_val in add_data_to_fold(X, y, fold, train_n):
    task = Task('binary')
    automl = TabularAutoML(
    task=task,
    timeout=9 * 3600,
    cpu_limit=os.cpu_count(),
    general_params={"use_algos": [['resnet']]},
    nn_params={
        "n_epochs": 5,
        "bs": 1024,
        "num_workers": 0,
        "path_to_save": None,
        "freeze_defaults": True,
        "cont_embedder": 'plr',
        'cat_embedder': 'weighted',
        "hidden_size": 32,
        'hid_factor': [32, 16],
        'embedding_size': 32,
        'stop_by_metric': True,
        'verbose_bar': True,
        "snap_params": {'k': 2, 'early_stopping': True, 'patience': 2, 'swa': True}
    },
    nn_pipeline_params={"use_qnt": False, "use_te": False},
    reader_params={'n_jobs': os.cpu_count(), 'cv': 7, 'random_state': 42, 'advanced_roles': True}
)

    # 학습
    automl.fit_predict(pd.concat([X_train, y_train], axis=1), roles={'target': 'Response'}, verbose=4)
    
    # 검증 점수 계산
    preds = automl.predict(X_val)
    auc = roc_auc_score(y_val, preds.data[:, 0])
    print(f"Fold {fold} AUC: {auc:.4f}")

# 최종 예측 및 결과 저장
test_pred = automl.predict(X_test)
y_pred = test_pred.data[:, 0]
pred_df = pd.DataFrame(y_pred, columns=['Response'])

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = os.path.join('./health_insurance/data/output', f'sample_submission_LightAutoML_{auc:.4f}_{timestamp}.csv')
df_submission = pd.read_csv('./health_insurance/data/sample_submission.csv')
df_submission['Response'] = pred_df
df_submission.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
