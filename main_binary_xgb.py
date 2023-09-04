'''
cls

time > output/main_binary_xgb-P-output.txt

python main_binary_xgb.py --label P >> output/main_binary_xgb-P-output.txt

time >> output/main_binary_xgb-P-output.txt


time > output/main_binary_xgb-V-output.txt

python main_binary_xgb.py --label V >> output/main_binary_xgb-V-output.txt

time >> output/main_binary_xgb-V-output.txt


'''
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, required=True)
args = parser.parse_args()
if not args.label in ['P','Q','R','S','T','V']:
    raise Exception("Invalid [label] argument: valid values = ['P','Q','R','S','T','V']")

random.seed(72)
cat_attribs = ['Fc0','Fc1','Fc2','Fc3','Fc4','Fc5','Fc6','Fc7','Fc8','Fc9','Fc10','Fc11']
noise_label = "035"
noise_ratio = 0.35
features = "24"

print("========================")
print("Noise (ratio):",noise_ratio)
print("========================")
dir_path = "datasets/main/"
train_dataset_csv = dir_path+'/'+'formulai-'+features+'-features-'+noise_label+'-noise-train.csv'
ttest_dataset_csv = dir_path+'/'+'formulai-'+features+'-features-'+noise_label+'-noise-test.csv'

train_df = pd.read_csv(train_dataset_csv)
ttest_df = pd.read_csv(ttest_dataset_csv)

# select tuning rows
# selRows = train_df[train_df['sample'].str.endswith("S3")].index

# extract binary labels
train_label_binary = train_df["label"]
ttest_label_binary = ttest_df["label"]
label_binary = {'P': 0 , 'Q': 0 , 'R': 0 , 'S': 0 , 'T': 0 , 'V': 0}
label_binary[args.label] = 1
train_label_binary = [label_binary[item] for item in train_label_binary]
ttest_label_binary = [label_binary[item] for item in ttest_label_binary]

# remove subject ID and label
# remove subject ID and label
train_df.drop(['sample','label'], axis=1, inplace=True)
ttest_df.drop(['sample','label'], axis=1, inplace=True)
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
encoder = full_pipeline.fit(train_df)

train_df, ttune_df, train_label_binary, ttune_label_binary = train_test_split(train_df, train_label_binary, test_size=0.2, random_state=72)

train_df = encoder.transform(train_df)
ttune_df = encoder.transform(ttune_df)
ttest_df = encoder.transform(ttest_df)

best_depth    = 0
best_f1_tune  = 0.0
best_th_tune  = 0.0
best_f1_test  = 0.0
best_Pr_test  = 0.0
best_Re_test  = 0.0

for DEPTH in range(10,31): # (10,12):
    print("------------------------")
    print("DEPTH:",DEPTH)
    print("------------------------")

    now = datetime.datetime.now()
    print("Time:",now.strftime("%H:%M:%S"))
    print("------------------------")

    #
    # Train
    # 
    model = xgb.XGBClassifier(booster='gbtree', random_state=72, max_depth=DEPTH)
    model.fit(train_df,train_label_binary)
    
    #
    # Tune
    # 
    ttune_pred = model.predict_proba(ttune_df)[:, 1]
    P, R, T = precision_recall_curve(ttune_label_binary, ttune_pred)
    for i in range(len(P)):
        if P[i] <= 0.00001 and R[i] <= 0.00001:
            P[i] = 1.0

    ttune_pred_auprc = np.round(ttune_pred,3)
    P2, R2, T2 = precision_recall_curve(ttune_label_binary, ttune_pred_auprc)
    for i in range(len(P2)):
        if P2[i] <= 0.00001 and R2[i] <= 0.00001:
            P2[i] = 1.0
    print(P2.tolist())
    print(R2.tolist())

    F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
    F1index = F1index[0]
    if F1index > 0:
        TH_avg = (T[F1index-1]+T[F1index])/2.0
    else:
        TH_avg = T[F1index]
    for idx in range(len(ttune_pred)):
        if ttune_pred[idx] < TH_avg:
            ttune_pred[idx] = 0
        else:
            ttune_pred[idx] = 1
    f1_tune = f1_score (ttune_label_binary , ttune_pred) # (?) .detach() )

    #
    # Test
    # 
    ttest_pred = model.predict_proba(ttest_df)[:, 1]
    for idx in range(len(ttest_pred)):
        if ttest_pred[idx] < 0.5:
            ttest_pred[idx] = 0
        else:
            ttest_pred[idx] = 1  
    f1_05 = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
    print("F1 (0.5):",f1_05)

    ttest_pred = model.predict_proba(ttest_df)[:, 1]
    for idx in range(len(ttest_pred)):
        if ttest_pred[idx] < TH_avg:
            ttest_pred[idx] = 0
        else:
            ttest_pred[idx] = 1  
    f1_test = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
    Pr = precision_score(ttest_label_binary, ttest_pred)
    Re = recall_score(ttest_label_binary, ttest_pred)
    print("F1 test:",f1_test)
    
    if f1_tune > best_f1_tune:
        best_depth   = DEPTH
        best_f1_tune = f1_tune
        best_th_tune = TH_avg
        best_f1_test = f1_test
        best_Pr_test = Pr
        best_Re_test = Re
        best_f1_05   = f1_05


print("------------------------")
print("Best:")
print("\t","Depth    :",best_depth)
print("\t","F1 tune  :",best_f1_tune)
print("\t","TH tune  :",best_th_tune)
print("\t","F1 test  :",best_f1_test)
print("\t","Precision:",best_Pr_test)
print("\t","Recall   :",best_Re_test)
print("========================")
