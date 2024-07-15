import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import logging
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

N_THREADS = os.cpu_count()
# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
np.random.seed(42)
torch.set_num_threads(N_THREADS)
if device == 'cuda':
    torch.cuda.set_device(0)

# Directory settings
dir = './health_insurance/data/'
output_dir = dir + "output/"
os.makedirs(output_dir, exist_ok=True)

# Logging setup
log_filename = os.path.join(output_dir, 'lightautoml_model_optimization.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(message)s')

# Start timer
start_time = time.time()

# Load data
df_train = pd.read_csv(dir + 'train_s.csv')
df_test = pd.read_csv(dir + 'test.csv')

# Define preprocessing function
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

# Apply preprocessing
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)

# Feature and label separation
X_train = df_train.drop(columns=['Response', 'id'])
y_train = df_train['Response']
X_test = df_test.drop(columns=['id'])

# Encode categorical variables
le = LabelEncoder()
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Apply SMOTE for handling imbalanced data
smote = SMOTE(random_state=42, sampling_strategy='minority')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create a new dataframe with resampled data
df_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
df_train_resampled['Response'] = y_train_resampled

# Split the resampled data into training and validation sets
# train_data, valid_data = train_test_split(df_train_resampled, test_size=0.2, random_state=42, stratify=df_train_resampled['Response'])

# LightAutoML model setup and training
task = Task('binary')
automl = TabularAutoML(
    task=task,
    timeout=9 * 3600,
    cpu_limit=os.cpu_count(),
    general_params={"use_algos": [['denselight']]},
    nn_params={
        "n_epochs": 5,
        "bs": 1024,
        "num_workers": 0,
        "path_to_save": None,
        "freeze_defaults": True,
        "cont_embedder": 'plr',
        'cat_embedder': 'weighted',
        'act_fun' : 'SiLU',
        "hidden_size": 32,
        'hid_factor': [512, 128],
        'embedding_size': 32,
        'stop_by_metric': True,
        'verbose_bar': True,
        "snap_params": {'k': 2, 'early_stopping': True, 'patience': 1, 'swa': True},
        'opt_params' : { 'lr' : 0.0003, 'weight_decay' : 0 }
    },
    nn_pipeline_params={"use_qnt": True, "use_te": False},
    reader_params={'n_jobs': os.cpu_count(), 'cv': 5, 'random_state': 42, 'advanced_roles': True}
)

# Train the model and get out-of-fold predictions
out_of_fold_predictions = automl.fit_predict(
    df_train_resampled, roles={'target': 'Response'}, verbose=4
)

# AUC evaluation
auc_score = roc_auc_score(df_train_resampled['Response'], out_of_fold_predictions.data)
logging.info(f"LightAutoML AUC score on training data: {auc_score:.4f}")
print(f"LightAutoML AUC score on training data: {auc_score:.4f}")

# Final test data prediction
test_pred = automl.predict(X_test)

# Convert probabilities to binary predictions
y_pred = test_pred.data[:, 0]
pred_df = pd.DataFrame(y_pred, columns=['Response'])

# Save results to sample_submission.csv
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file_path = os.path.join(output_dir, f'sample_submission_LightAutoML_{auc_score:.4f}_{timestamp}.csv')
df_submission = pd.read_csv(dir + 'sample_submission.csv')
df_submission['Response'] = pred_df
df_submission.to_csv(output_file_path, index=False)

logging.info(f"Predictions saved to {output_file_path}")
print(f"Predictions saved to {output_file_path}")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Elapsed time: {elapsed_time:.2f} seconds")
