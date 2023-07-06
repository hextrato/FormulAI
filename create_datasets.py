'''
python create_datasets.py
'''
import pandas as pd
import os
import pickle
import random

from hextrato import formulai as fai

#
# generate dataset variations based on number-of-features 
#

print ("Generating datasets...")

# # generate dataset variations based on number-of-features 
# 
# random.seed(72)
#
# features = [1,2,4,8] # *2
# sample_size = 8
# test_size = 2
# 
# for idx in range(len(features)):
#     f = features[idx] * 2
#     print ("Features:",f)
#     dir_path = "datasets/features/"
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     faigen = fai.FAIGenerator()
#     faigen.set_random_noise(0.0)
#     faigen.set_attributes(categorical=features[idx],continuous=features[idx])
#     faigen.set_sample_size(min_sample=sample_size,max_sample=sample_size)
#     faigen.set_test_sample_size(test_size)
#     faigen.generate()
#     train,test = faigen.dataframes()
#     train.to_csv(dir_path+'/'+'formulai-'+(str(f).zfill(2))+'-features-train.csv', index=False)
#     test.to_csv(dir_path+'/'+'formulai-'+(str(f).zfill(2))+'-features-test.csv', index=False)
# 
# # generate dataset variations based on noise
# 
# random.seed(72)
#
# features = 4
# sample_size = 15
# test_size = 3
# noise = {"000":0.000,"025":0.025,"050":0.050,"075":0.075,"100":0.100,"150":0.150,"200":0.200,"250":0.250}
# 
# for noise_label in noise:
#     noise_ratio = noise[noise_label]
#     print ("Noise (ratio):",noise_ratio)
#     dir_path = "datasets/noise/"
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     faigen = fai.FAIGenerator()
#     faigen.set_random_noise(noise_ratio)
#     faigen.set_attributes(categorical=features,continuous=features)
#     faigen.set_sample_size(min_sample=sample_size,max_sample=sample_size)
#     faigen.set_test_sample_size(test_size)
#     faigen.generate()
#     train,test = faigen.dataframes()
#     train.to_csv(dir_path+'/'+'formulai-'+(noise_label.zfill(2))+'-noise-train.csv', index=False)
#     test.to_csv(dir_path+'/'+'formulai-'+(noise_label.zfill(2))+'-noise-test.csv', index=False)

# generate main dataset

random.seed(72)

features = 12
sample_size = 18
test_size = 4
noise_label = "035"
noise_ratio = 0.35

dir_path = "datasets/main/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
faigen = fai.FAIGenerator()
faigen.set_random_noise(noise_ratio)
faigen.set_attributes(categorical=features,continuous=features)
faigen.set_sample_size(min_sample=sample_size,max_sample=sample_size)
faigen.set_test_sample_size(test_size)
faigen.generate()
train,test = faigen.dataframes()
train.to_csv(dir_path+'/'+'formulai-'+(str(features*2).zfill(2))+'-features'+noise_label+'-noise-train.csv', index=False)
test.to_csv(dir_path+'/'+'formulai-'+(str(features*2).zfill(2))+'-features'+noise_label+'-noise-test.csv', index=False)

print ("Done!")
