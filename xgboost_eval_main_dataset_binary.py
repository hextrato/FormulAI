'''
python xgboost_eval_main_dataset_binary.py > output/xgboost_eval_main_dataset_binary-output.txt
'''
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

random.seed(471582693)

# make dir
dir_path = "datasets/main/"
train_dataset_csv = dir_path+"formulai-train.csv"
ttest_dataset_csv = dir_path+"formulai-test.csv"

train_df = pd.read_csv(train_dataset_csv)
ttest_df = pd.read_csv(ttest_dataset_csv)

# select tuning rows
# selRows = train_df[train_df['sample'].str.endswith("S3")].index

# extract labels
# train_label = train_df["label"]
# ttest_label = ttest_df["label"]
# train_label = pd.Categorical(train_label).codes
# ttest_label = pd.Categorical(ttest_label).codes

# extract binary labels
train_label_binary = train_df["label"]
ttest_label_binary = ttest_df["label"]
label_binary = {'P': 0 , 'Q': 0 , 'R': 0 , 'S': 0 , 'T': 0 , 'V': 1}
train_label_binary = [label_binary[item] for item in train_label_binary]
ttest_label_binary = [label_binary[item] for item in ttest_label_binary]

# remove subject ID and label
# remove subject ID and label
train_df.drop(['sample','label'], axis=1, inplace=True)
ttest_df.drop(['sample','label'], axis=1, inplace=True)
cat_attribs = ['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15']
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
encoder = full_pipeline.fit(train_df)

train_df, ttune_df, train_label_binary, ttune_label_binary = train_test_split(train_df, train_label_binary, test_size=0.1, random_state=72)

train_df = encoder.transform(train_df)
ttune_df = encoder.transform(ttune_df)
ttest_df = encoder.transform(ttest_df)

best_depth    = 0
best_f1_tune  = 0.0
best_th_tune  = 0.0
best_f1_test  = 0.0

for DEPTH in range(3,25): # (10,12):
    print("------------------------")
    print("DEPTH:",DEPTH)
    print("------------------------")

    #
    # Train
    # 
    model = xgb.XGBRegressor(booster='gbtree', random_state=72, max_depth=DEPTH)
    # model = xgb.XGBRegressor(max_depth=DEPTH)
    model.fit(train_df,train_label_binary)
    
    #
    # Tune
    # 
    ttune_pred = model.predict(ttune_df)
    P, R, T = precision_recall_curve(ttune_label_binary, ttune_pred)
    for i in range(len(P)):
        if P[i] <= 0.00001 and R[i] <= 0.00001:
            P[i] = 1.0
    # print(P)
    # print(R)
    F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
    for idx in range(len(ttune_pred)):
        if ttune_pred[idx] < T[F1index]:
            ttune_pred[idx] = 0
        else:
            ttune_pred[idx] = 1
    f1_tune = f1_score (ttune_label_binary , ttune_pred) # (?) .detach() )

    #
    # Test
    # 
    ttest_pred = model.predict(ttest_df)
    for idx in range(len(ttest_pred)):
        if ttest_pred[idx] < 0.5:
            ttest_pred[idx] = 0
        else:
            ttest_pred[idx] = 1  
    f1_05 = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
    print("F1 (0.5):",f1_05)

    ttest_pred = model.predict(ttest_df)
    for idx in range(len(ttest_pred)):
        if ttest_pred[idx] < T[F1index]:
            ttest_pred[idx] = 0
        else:
            ttest_pred[idx] = 1  
    f1_test = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
    print("F1 test:",f1_test)
    
    if f1_tune > best_f1_tune:
        best_depth   = DEPTH
        best_f1_tune = f1_tune
        best_th_tune = T[F1index]
        best_f1_test = f1_test
        best_f1_05   = f1_05


print("------------------------")
print("Best:")
#print("\t","Accuracy:",best_accuracy)
print("\t","Depth:",best_depth)
print("\t","F1 tune:",best_f1_tune)
print("\t","TH tune:",best_th_tune)
print("\t","F1 test:",best_f1_test)
print("\t","F1 test (0.5):",best_f1_05)
print("========================")

