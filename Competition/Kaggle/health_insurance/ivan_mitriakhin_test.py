import numpy as np
import pandas as pd
import polars as pl

orig_train = pl.read_csv('./health_insurance/data/train_n.csv')
print("Original train shape",orig_train.shape)

train = pl.read_csv('./health_insurance/data/train.csv')
print("Train shape",train.shape)

train = pl.concat([train, orig_train])
print("Train + orig_train shape",train.shape)

test = pl.read_csv('./health_insurance/data/test.csv')

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


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import catboost as cb
from catboost import Pool
import gc

FOLDS = 2
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
test_preds = np.zeros((len(X_test), FOLDS), dtype=np.float32)
scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print("#" * 25)
    print(f"# Fold {fold + 1}")
    print("#" * 25)
    X_train_fold = X.loc[train_idx]
    y_train_fold = y.loc[train_idx].values
    X_val_fold = X.loc[val_idx]
    y_val_fold = y.loc[val_idx].values
    X_train_pool = Pool(X_train_fold, y_train_fold, cat_features=X.columns.values)
    X_val_pool = Pool(X_val_fold, y_val_fold, cat_features=X.columns.values)
    X_test_pool = Pool(X_test, cat_features=X_test.columns.values)
    
    # Train CatBoost
    model = cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        class_names=[0, 1],
        learning_rate=0.075,
        iterations=5000,
        depth=9,
        random_strength=0,
        l2_leaf_reg=0.5,
        max_leaves=512,
        fold_permutation_block=64,
        task_type='GPU',
        random_seed=42,
        verbose=False
    )

    model.fit(X=X_train_pool, 
              eval_set=X_val_pool, 
              verbose=500, 
              early_stopping_rounds=200)

    score = model.best_score_['validation']['AUC']
    print('Fold ROC-AUC score: ', score)
    scores.append(score)

    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]
    
    del X_train_fold, y_train_fold
    del X_val_fold, y_val_fold
    del X_train_pool, X_val_pool, X_test_pool
    del model
    gc.collect()

print('Mean ROC-AUC score: ', sum(scores) / FOLDS)

test_preds = np.mean(test_preds, axis=1)
submission = pd.read_csv('./health_insurance/data/sample_submission.csv')
submission[target] = test_preds.astype(np.float32)
submission['id'] = submission['id'].astype(np.int32)
submission.to_csv('submission_test.csv', index=False)