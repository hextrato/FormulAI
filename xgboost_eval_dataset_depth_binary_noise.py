'''
python xgboost_eval_dataset_depth_binary_noise.py --noise 0 > output/xgboost_eval_dataset_depth_binary_noise-0-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 1 > output/xgboost_eval_dataset_depth_binary_noise-1-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 2 > output/xgboost_eval_dataset_depth_binary_noise-2-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 3 > output/xgboost_eval_dataset_depth_binary_noise-3-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 4 > output/xgboost_eval_dataset_depth_binary_noise-4-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 5 > output/xgboost_eval_dataset_depth_binary_noise-5-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 6 > output/xgboost_eval_dataset_depth_binary_noise-6-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 7 > output/xgboost_eval_dataset_depth_binary_noise-7-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 8 > output/xgboost_eval_dataset_depth_binary_noise-8-output.txt
python xgboost_eval_dataset_depth_binary_noise.py --noise 9 > output/xgboost_eval_dataset_depth_binary_noise-9-output.txt
'''
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--noise', type=int, required=True)
args = parser.parse_args()
# print("noise",args.noise)
if args.noise < 0 or args.noise > 9:
    raise Exception("Invalid [noise] argument: valid values [0,9]")

random.seed(963258741)

NOISE = args.noise

# make dir
dir_path = "datasets/eval/"+str(NOISE)+"/"
train_dataset_csv = dir_path+"formula"+str(NOISE)+"-train.csv"
ttest_dataset_csv = dir_path+"formula"+str(NOISE)+"-test.csv"

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
cat_attribs = ['F0','F1','F2','F3','F4','F5','F6','F7']
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
encoder = full_pipeline.fit(train_df)
train_df = encoder.transform(train_df)
ttest_df = encoder.transform(ttest_df)

best_accuracy = 0.0
best_depth    = 0
best_f1       = 0.0

print("========================")
print("NOISE:",NOISE)
print("========================")

for DEPTH in range(6,16):
    print("------------------------")
    print("DEPTH:",DEPTH)
    print("------------------------")

    #
    # Train
    # 
    model = xgb.XGBRegressor(booster='gbtree', random_state=72, max_depth=DEPTH)
    model.fit(train_df,train_label_binary)
    train_pred = model.predict(train_df)
    P, R, T = precision_recall_curve(train_label_binary, train_pred)
    for i in range(len(P)):
        if P[i] <= 0.00001 and R[i] <= 0.00001:
            P[i] = 1.0
    #print(P)
    #print(R)
    #print(T)
    F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
    for idx in range(len(train_pred)):
        if train_pred[idx] < T[F1index]:
            train_pred[idx] = 0
        else:
            train_pred[idx] = 1
    #scores = cross_val_score (model,train_df,train_label,cv=5)
    #accuracy = np.round(scores,4)
    #mean_accuracy = np.round(scores.mean(),4)
    #print("Accuracy:",accuracy)
    #print("Accuracy (mean):",mean_accuracy)

    #
    # Test
    # 
    ttest_pred = model.predict(ttest_df)
    for idx in range(len(ttest_pred)):
        if ttest_pred[idx] < T[F1index]:
            ttest_pred[idx] = 0
        else:
            ttest_pred[idx] = 1  
    f1 = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
    print("F1:",f1)
    
    if f1 > best_f1:
        #best_accuracy = mean_accuracy
        best_depth = DEPTH
        best_f1    = f1

print("------------------------")
print("Best:")
#print("\t","Accuracy:",best_accuracy)
print("\t","Depth:",best_depth)
print("\t","F1:",best_f1)
print("========================")

