'''
python xgboost_eval_dataset_depth_mclass_noise.py --noise 0 > output/xgboost_eval_dataset_depth_mclass_noise-0-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 1 > output/xgboost_eval_dataset_depth_mclass_noise-1-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 2 > output/xgboost_eval_dataset_depth_mclass_noise-2-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 3 > output/xgboost_eval_dataset_depth_mclass_noise-3-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 4 > output/xgboost_eval_dataset_depth_mclass_noise-4-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 5 > output/xgboost_eval_dataset_depth_mclass_noise-5-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 6 > output/xgboost_eval_dataset_depth_mclass_noise-6-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 7 > output/xgboost_eval_dataset_depth_mclass_noise-7-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 8 > output/xgboost_eval_dataset_depth_mclass_noise-8-output.txt
python xgboost_eval_dataset_depth_mclass_noise.py --noise 9 > output/xgboost_eval_dataset_depth_mclass_noise-9-output.txt
'''
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score 
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
train_label = train_df["label"]
ttest_label = ttest_df["label"]
train_label = pd.Categorical(train_label).codes
ttest_label = pd.Categorical(ttest_label).codes

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
best_micro_f1 = 0.0
best_macro_f1 = 0.0

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
    model = xgb.XGBClassifier(booster='gbtree', objective='multi:softprob', random_state=72, max_depth=DEPTH)
    model.fit(train_df,train_label)
    #scores = cross_val_score (model,train_df,train_label,cv=5)
    #accuracy = np.round(scores,4)
    #mean_accuracy = np.round(scores.mean(),4)
    #print("Accuracy:",accuracy)
    #print("Accuracy (mean):",mean_accuracy)

    #
    # Test
    # 
    ttest_pred = model.predict(ttest_df)
    micro_f1 = f1_score(ttest_label, ttest_pred,average="micro")
    print("micro f1:",micro_f1)
    macro_f1 = f1_score(ttest_label, ttest_pred,average="macro")
    print("macro f1:",macro_f1)
    
    cm = confusion_matrix (ttest_label, ttest_pred)
    print("Confusion matrix:")
    print(cm)

    if macro_f1 > best_macro_f1:
        #best_accuracy = mean_accuracy
        best_depth    = DEPTH
        best_micro_f1 = micro_f1
        best_macro_f1 = macro_f1

print("------------------------")
print("Best:")
#print("\t","Accuracy:",best_accuracy)
print("\t","Depth:",best_depth)
print("\t","Micro F1:",best_micro_f1)
print("\t","Macro F1:",best_macro_f1)
print("========================")

