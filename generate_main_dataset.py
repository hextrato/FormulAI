import pandas as pd
import os
import pickle
import random

from hextrato import formulai as fai

random.seed(246813579)

noise_ratio = 0.75

# make dir
dir_path = "datasets/main"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

faigen = fai.FAIGenerator()
faigen.set_random_noise(noise_ratio)
faigen.set_attributes(categorical=16,continuous=16)
faigen.set_sample_size(min_sample=8,max_sample=12)
faigen.set_test_sample_size(3)
faigen.generate()
#fai.json()
train,test = faigen.dataframes()
    
#fileObj = open(dir_path+'/'+'formulai-dataframe-train.pickle', 'wb')
#pickle.dump(train,fileObj)
#fileObj.close()

#fileObj = open(dir_path+'/'+'formulai-dataframe-test.pickle', 'wb')
#pickle.dump(test,fileObj)
#fileObj.close()

train.to_csv(dir_path+'/'+'formulai-train.csv', index=False)
test.to_csv(dir_path+'/'+'formulai-test.csv', index=False)
