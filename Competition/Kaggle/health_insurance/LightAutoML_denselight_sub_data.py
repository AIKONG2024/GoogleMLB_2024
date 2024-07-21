import numpy as np
import pandas as pd
import polars as pl
import logging
from sklearn.metrics import roc_auc_score
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from datetime import datetime
import time
import os
import torch
from sklearn.model_selection import train_test_split

N_THREADS = os.cpu_count()
# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
np.random.seed(42)
torch.set_num_threads(N_THREADS)
if device == 'cuda':
    torch.cuda.set_device(0)
    
start_time = time.time()
orig_train = pl.read_csv('./health_insurance/data/train.csv')
print("Original train shape", orig_train.shape)

train = pl.read_csv('./health_insurance/data/train_n.csv')
print("Train shape", train.shape)

train = pl.concat([train, orig_train])
print("Train + orig_train shape", train.shape)

test = pl.read_csv('./health_insurance/data/test.csv')
print("Test shape", test.shape)

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
    
X = train.drop(['id', target], axis=1)
y = train[target]
X_test = test.drop(['id', target], axis=1)

# Define memory reduction function
def reduce_mem_usage(df):
    """ Reduce memory usage of dataframe by modifying data types. """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('object')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# Apply memory reduction
df_train = reduce_mem_usage(train)
df_test = reduce_mem_usage(test)

# Feature and label separation
X_train = df_train.drop(columns=['Response', 'id'])
y_train = df_train['Response']
X_test = df_test.drop(columns=['id'])

# Create a new dataframe with resampled data
df_train_resampled = pd.DataFrame(X_train, columns=X_train.columns)
df_train_resampled['Response'] = y_train

X_train, X_val = train_test_split(df_train_resampled, test_size=0.2, random_state=42, shuffle=True, stratify=df_train_resampled.Response)

# LightAutoML model setup and training
task = Task('binary')
automl = TabularAutoML(
    task = task, 
    timeout = 600 * 3600,
    cpu_limit = 2,
    gpu_ids = '0',
    general_params = {"use_algos": [['denselight']]},
    nn_params = {
        "n_epochs": 10, 
        "bs": 1024, 
        "num_workers": 0, 
        "path_to_save": None, 
        "freeze_defaults": True,
        "cont_embedder": 'plr',
        'cat_embedder': 'weighted',
        'act_fun': 'SiLU',
        "hidden_size": [512, 128],
        'stop_by_metric': True,
        'embedding_size': 32,
        'verbose_bar': True,
        "snap_params": { 'k': 2, 'early_stopping': True, 'patience': 1, 'swa': True }, 
        'opt_params': { 'lr': 0.0003 , 'weight_decay': 0 }
    },
    nn_pipeline_params = {"use_qnt": True, "use_te": False},
    reader_params = {'n_jobs': 12, 'cv': 12, 'random_state': 42, 'advanced_roles': True}
)

# 전체 훈련 데이터로 모델 학습
automl.fit_predict(
    X_train, 
    roles = {
        'target': 'Response',
        'drop': ['id'],
    },
    verbose = 4
)

# 검증 데이터에 대한 예측
val_pred = automl.predict(X_val)

# AUC 평가
if val_pred.data.shape[1] == 1:
    auc_score = roc_auc_score(X_val['Response'], val_pred.data)
else:
    auc_score = roc_auc_score(X_val['Response'], val_pred.data[:, 1])

print(f"LightAutoML AUC score on validation data: {auc_score:.4f}")

test_pred = automl.predict(X_test)

if test_pred.data.shape[1] == 1:
    y_pred = test_pred.data.flatten()
else:
    y_pred = test_pred.data[:, 1]

pred_df = pd.DataFrame(y_pred, columns=['Response'])

# Save results to sample_submission.csv
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = os.path.join('./health_insurance/data/output', f'sample_submission_LightAutoML_{auc_score:.4f}_{timestamp}.csv')
df_submission = pd.read_csv('./health_insurance/data/sample_submission.csv')
df_submission['Response'] = pred_df
df_submission.to_csv(output_file_path, index=False)

logging.info(f"Predictions saved to {output_file_path}")
print(f"Predictions saved to {output_file_path}")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Elapsed time: {elapsed_time:.2f} seconds")
