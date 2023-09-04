'''
python eval_feature_mlabel_xgb.py > output/eval_feature_mlabel_xgb-output.txt

python eval_feature_mlabel_xgb.py > output/eval_feature_mlabel_xgb-output-32.txt
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

random.seed(72)
# features = [1,2,4,8]
features = [16]
# cat_attribs_list = [ ['Fc0'] , ['Fc0','Fc1'] , ['Fc0','Fc1','Fc2','Fc3'] , ['Fc0','Fc1','Fc2','Fc3','Fc4','Fc5','Fc6','Fc7'] , ['Fc0','Fc1','Fc2','Fc3','Fc4','Fc5','Fc6','Fc7','Fc8','Fc9','Fc10','Fc11','Fc12','Fc13','Fc14','Fc15'] ]
cat_attribs_list = [ ['Fc0','Fc1','Fc2','Fc3','Fc4','Fc5','Fc6','Fc7','Fc8','Fc9','Fc10','Fc11','Fc12','Fc13','Fc14','Fc15'] ]

for idx in range(len(features)):
    f = features[idx] * 2
    print("========================")
    print ("Features:",f)
    print("========================")
    dir_path = "datasets/eval_features/"
    train_dataset_csv = dir_path+'/'+'formulai-'+(str(f).zfill(2))+'-features-train.csv'
    ttest_dataset_csv = dir_path+'/'+'formulai-'+(str(f).zfill(2))+'-features-test.csv'

    train_df = pd.read_csv(train_dataset_csv)
    ttest_df = pd.read_csv(ttest_dataset_csv)

    # extract labels
    train_label = train_df["label"]
    ttest_label = ttest_df["label"]
    train_label = pd.Categorical(train_label).codes
    ttest_label = pd.Categorical(ttest_label).codes

    # remove subject ID and label
    train_df.drop(['sample','label'], axis=1, inplace=True)
    ttest_df.drop(['sample','label'], axis=1, inplace=True)
    # cat_attribs = ['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15']
    cat_attribs = cat_attribs_list[idx]
    full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
    encoder = full_pipeline.fit(train_df)
    train_df = encoder.transform(train_df)
    ttest_df = encoder.transform(ttest_df)

    best_depth    = 0
    best_f1_test  = 0.0

    for DEPTH in range(5,21):
        print("------------------------")
        print("DEPTH:",DEPTH)
        print("------------------------")

        #
        # Train
        # 
        model = xgb.XGBClassifier(booster='gbtree', random_state=72, max_depth=DEPTH)
        model.fit(train_df,train_label)

        #
        # Test
        # 
        ttest_pred = model.predict(ttest_df)
        f1_test = f1_score (ttest_label , ttest_pred, average="weighted")
        print("F1 test:",f1_test)

        if f1_test > best_f1_test:
            best_depth   = DEPTH
            best_f1_test = f1_test

        cm = confusion_matrix (ttest_label, ttest_pred)
        print("Confusion matrix:")
        print(cm)

    print("------------------------")
    print("Best:")
    print("\t","Depth:",best_depth)
    print("\t","F1 test:",best_f1_test)
    print("========================")

