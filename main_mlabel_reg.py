'''
cls

time > output/main_mlabel_reg-output.txt

python main_mlabel_reg.py >> output/main_mlabel_reg-output.txt

time >> output/main_mlabel_reg-output.txt


'''
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

# extract labels
train_label = train_df["label"]
ttest_label = ttest_df["label"]
train_label = pd.Categorical(train_label).codes
ttest_label = pd.Categorical(ttest_label).codes

# remove subject ID and label
train_df.drop(['sample','label'], axis=1, inplace=True)
ttest_df.drop(['sample','label'], axis=1, inplace=True)
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)], remainder='passthrough')
encoder = full_pipeline.fit(train_df)
train_df = encoder.transform(train_df)
ttest_df = encoder.transform(ttest_df)

#
# Train
# 
model = LogisticRegression(max_iter=100000, class_weight='balanced')
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

