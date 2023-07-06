'''
python eval_feature_mlabel_svm.py > output/eval_feature_mlabel_svm-output.txt
'''
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

random.seed(72)
features = [1,2,4,8]
cat_attribs_list = [ ['F0'] , ['F0','F1'] , ['F0','F1','F2','F3'] , ['F0','F1','F2','F3','F4','F5','F6','F7'] ]

for idx in range(len(features)):
    f = features[idx] * 2
    print("========================")
    print ("Features:",f)
    print("========================")
    dir_path = "datasets/features/"
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

    #
    # Train
    # 
    model = SVC(max_iter=100000,gamma='scale')
    model.fit(train_df,train_label)

    #
    # Test
    # 
    ttest_pred = model.predict(ttest_df)
    f1_test = f1_score (ttest_label , ttest_pred, average="weighted")
    print("F1 test:",f1_test)

    #if f1_test > best_f1_test:
    # best_f1_tune = f1_tune
    # best_th_tune = T[F1index]
    #    best_f1_test = f1_test

    cm = confusion_matrix (ttest_label, ttest_pred)
    print("Confusion matrix:")
    print(cm)

    print("------------------------")
    print("Best:")
    print("\t","F1 test:",f1_test)
    print("========================")

