import pandas as pd
import os
import pickle
import random

from hextrato import formulai as fai

random.seed(135792468)

noise = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for noise_idx in range(len(noise)):
    # make dir
    dir_path = "datasets/eval/"+str(noise_idx)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # dataset [0,9] with noise_ratio [0.0%,90.0%]
    noise_ratio = noise[noise_idx]
    faigen = fai.FAIGenerator()
    faigen.set_random_noise(noise_ratio)
    faigen.generate()
    #fai.json()
    train,test = faigen.dataframes()
        
    #fileObj = open(dir_path+'/'+'formula'+str(noise_idx)+'-dataframe-train.pickle', 'wb')
    #pickle.dump(train,fileObj)
    #fileObj.close()

    #fileObj = open(dir_path+'/'+'formula'+str(noise_idx)+'-dataframe-test.pickle', 'wb')
    #pickle.dump(test,fileObj)
    #fileObj.close()

    train.to_csv(dir_path+'/'+'formula'+str(noise_idx)+'-train.csv', index=False)
    test.to_csv(dir_path+'/'+'formula'+str(noise_idx)+'-test.csv', index=False)
