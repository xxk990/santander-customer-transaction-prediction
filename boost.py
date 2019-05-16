import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import gc
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from numba import jit
from sklearn import preprocessing
train_df = pd.read_csv("train3.csv")
feature1 = train_df.columns.values.tolist()
test_df = pd.read_csv("test3.csv")
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']
train_df = preprocessing.scale(train_df[features])
train_df = pd.DataFrame(train_df,columns = features)
param2 = {
        'bagging_freq': 5,
        'bagging_fraction': 0.38,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.04,
        'learning_rate': 0.01,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1
    }

from sklearn.metrics import roc_auc_score, roc_curve
skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2319)
oof = np.zeros(len(train_df))
feature_importance_df = pd.DataFrame()
predictions = np.zeros(len(test_df))
for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param2, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    #predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / 5
print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("lgb_submission(13).csv", index=False)




##############################################################################################################################################
from catboost import Pool, CatBoostClassifier
model = CatBoostClassifier(loss_function="Logloss",
                               eval_metric="AUC",
                               task_type="GPU",
                               learning_rate=0.02,
                               iterations=15000000,
                               random_seed=42,
                               l2_leaf_reg=0.28,
                               od_type="Iter",
                               depth=2,
                               use_best_model=True)

y_valid_pred = 0 * target
y_test_pred = 0
y_train_pred = 0
sum_score = 0
for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):
    _train = Pool(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    _valid = Pool(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    #_train = Pool(X_train, label=y_train)
    #_valid = Pool(X_valid, label=y_valid)
    print("fold n°{}".format(fold_))
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=2000,
                          early_stopping_rounds = 6000
                         )
    pred = fit_model.predict_proba(train_df.iloc[val_idx][features])[:,1]
    print( "  auc = ", roc_auc_score(target.iloc[val_idx], pred) )
    sum_score = sum_score+roc_auc_score(target.iloc[val_idx], pred)
    y_valid_pred.iloc[val_idx] = pred
    y_train_pred += pd.DataFrame(fit_model.predict_proba(train_df[features])[:,1])
    y_test_pred += pd.DataFrame(fit_model.predict_proba(test_df[features])[:,1])
#y_test_pred *= 12
print('all auc=', sum_score/12)
#y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred= pd
y_test_pred.to_csv("cat_feature_test.csv", index=False)
y_train_pred.to_csv("cat_feature_train.csv", index=False)