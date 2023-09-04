'''
python eval_noise_binary_svm.py --label P > output/eval_noise_binary_svm-P-output.txt
python eval_noise_binary_svm.py --label V > output/eval_noise_binary_svm-V-output.txt

python eval_noise_binary_svm.py --label P > output/eval_noise_binary_svm-P-output_03x.txt
python eval_noise_binary_svm.py --label V > output/eval_noise_binary_svm-V-output_03x.txt
'''
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, required=True)
args = parser.parse_args()
if not args.label in ['P','Q','R','S','T','V']:
    raise Exception("Invalid [label] argument: valid values = ['P','Q','R','S','T','V']")

random.seed(72)
cat_attribs = ['Fc0','Fc1','Fc2','Fc3']
# noise = {"000":0.000,"025":0.025,"050":0.050,"075":0.075,"100":0.100,"150":0.150,"200":0.200,"250":0.250}
noise = {"300":0.300,"350":0.350}

for noise_label in noise:
    noise_ratio = noise[noise_label]
    print("========================")
    print("Noise (ratio):",noise_ratio)
    print("========================")
    dir_path = "datasets/eval_noise/"
    train_dataset_csv = dir_path+'/'+'formulai-'+noise_label+'-noise-train.csv'
    ttest_dataset_csv = dir_path+'/'+'formulai-'+noise_label+'-noise-test.csv'

    train_df = pd.read_csv(train_dataset_csv)
    ttest_df = pd.read_csv(ttest_dataset_csv)

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

    #
    # Train
    # 
    model = SVC(max_iter=200000,gamma='scale',probability=True)
    model.fit(train_df,train_label_binary)

    #
    # Tune
    # 
    ttune_pred = model.predict_proba(ttune_df)[:, 1]
    P, R, T = precision_recall_curve(ttune_label_binary, ttune_pred)
    for i in range(len(P)):
        if P[i] <= 0.00001 and R[i] <= 0.00001:
            P[i] = 1.0
    # print(P)
    # print(R)
    F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
    F1index = F1index[0]
    if F1index > 0:
        TH_avg = (T[F1index-1]+T[F1index])/2.0
    else:
        TH_avg = T[F1index]
    print("F1index",F1index)
    print("TH_avg",TH_avg)
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
    Pt = precision_score(ttest_label_binary, ttest_pred)
    Rt = recall_score(ttest_label_binary, ttest_pred)
    f1_test = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
    print("F1 test:",f1_test)
    print("Precision", Pt)
    print("Recall", Rt)

    print("------------------------")
    print("Scores:")
    print("\t","F1 test  :",f1_05,"(th = 0.5)")
    print("\t","F1 tune  :",f1_tune)
    print("\t","TH tune  :",TH_avg)
    print("\t","F1 test  :",f1_test)
    print("\t","Precision:",Pt)
    print("\t","Recall   :",Rt)
    print("========================")



# '''
# python main_dataset_svm_binary.py > output/main_dataset_svm_binary-output.txt
# '''
# import argparse
# import pandas as pd
# import numpy as np
# from sklearn.svm import LinearSVC
# import random
# from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_recall_curve
# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# 
# random.seed(471582693)
# 
# # make dir
# dir_path = "datasets/main/"
# train_dataset_csv = dir_path+"formulai-train.csv"
# ttest_dataset_csv = dir_path+"formulai-test.csv"
# 
# train_df = pd.read_csv(train_dataset_csv)
# ttest_df = pd.read_csv(ttest_dataset_csv)
# 
# # select tuning rows
# # selRows = train_df[train_df['sample'].str.endswith("S3")].index
# 
# # extract labels
# # train_label = train_df["label"]
# # ttest_label = ttest_df["label"]
# # train_label = pd.Categorical(train_label).codes
# # ttest_label = pd.Categorical(ttest_label).codes
# 
# # extract binary labels
# train_label_binary = train_df["label"]
# ttest_label_binary = ttest_df["label"]
# label_binary = {'P': 0 , 'Q': 0 , 'R': 0 , 'S': 0 , 'T': 0 , 'V': 1}
# train_label_binary = [label_binary[item] for item in train_label_binary]
# ttest_label_binary = [label_binary[item] for item in ttest_label_binary]
# 
# # remove subject ID and label
# # remove subject ID and label
# train_df.drop(['sample','label'], axis=1, inplace=True)
# ttest_df.drop(['sample','label'], axis=1, inplace=True)
# cat_attribs = ['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15']
# full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
# encoder = full_pipeline.fit(train_df)
# 
# train_df, ttune_df, train_label_binary, ttune_label_binary = train_test_split(train_df, train_label_binary, test_size=0.1, random_state=72)
# 
# train_df = encoder.transform(train_df)
# ttune_df = encoder.transform(ttune_df)
# ttest_df = encoder.transform(ttest_df)
# 
# best_f1_tune  = 0.0
# best_th_tune  = 0.0
# best_f1_test  = 0.0
# 
# #
# # Train
# # 
# model = LinearSVC(max_iter=100000)
# model.fit(train_df,train_label_binary)
# 
# #
# # Tune
# # 
# ttune_pred = model.predict(ttune_df)
# P, R, T = precision_recall_curve(ttune_label_binary, ttune_pred)
# for i in range(len(P)):
#     if P[i] <= 0.00001 and R[i] <= 0.00001:
#         P[i] = 1.0
# # print(P)
# # print(R)
# F1index, = np.where( (2*(P*R)/(P+R)) == max((2*(P*R)/(P+R))))
# for idx in range(len(ttune_pred)):
#     if ttune_pred[idx] < T[F1index][0]:
#         ttune_pred[idx] = 0
#     else:
#         ttune_pred[idx] = 1
# f1_tune = f1_score (ttune_label_binary , ttune_pred) # (?) .detach() )
# 
# #
# # Test
# # 
# ttest_pred = model.predict(ttest_df)
# for idx in range(len(ttest_pred)):
#     if ttest_pred[idx] < 0.5:
#         ttest_pred[idx] = 0
#     else:
#         ttest_pred[idx] = 1  
# f1_05 = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
# print("F1 (0.5):",f1_05)
# 
# ttest_pred = model.predict(ttest_df)
# for idx in range(len(ttest_pred)):
#     if ttest_pred[idx] < T[F1index][0]:
#         ttest_pred[idx] = 0
#     else:
#         ttest_pred[idx] = 1  
# f1_test = f1_score (ttest_label_binary , ttest_pred) # (?) .detach() )
# print("F1 test:",f1_test)
# 
# if f1_tune > best_f1_tune:
#     best_f1_tune = f1_tune
#     best_th_tune = T[F1index][0]
#     best_f1_test = f1_test
#     best_f1_05   = f1_05
# 
# 
# print("------------------------")
# print("Best:")
# #print("\t","Accuracy:",best_accuracy)
# print("\t","F1 tune:",best_f1_tune)
# print("\t","TH tune:",best_th_tune)
# print("\t","F1 test:",best_f1_test)
# print("\t","F1 test (0.5):",best_f1_05)
# print("========================")

