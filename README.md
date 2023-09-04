# FormulAI

Python package to generate synthetic hybrid datasets (including categorical and continuous features). 

## Project Organization

```
│   requirements.txt                   <- Required Python packages
│   create_datasets.py                 <- Python script to generate the evaluation datasets
│   eval_feature_binary_reg.py         <- Python script to evaluate impact of features using Logistic Regression in a binary classification
│   eval_feature_binary_svm.py         <- Python script to evaluate impact of features using SVM in a binary classification
│   eval_feature_binary_xgb.py         <- Python script to evaluate impact of features using XGBoost in a binary classification
│   eval_feature_mlabel_reg.py         <- Python script to evaluate impact of features using Logistic Regression in a multilabel classification
│   eval_feature_mlabel_svm.py         <- Python script to evaluate impact of features using SVM in a multilabel classification
│   eval_feature_mlabel_xgb.py         <- Python script to evaluate impact of features using XGBoost in a multilabel classification
│   eval_noise_binary_reg.py           <- Python script to evaluate impact of noise using Logistic Regression in a binary classification
│   eval_noise_binary_svm.py           <- Python script to evaluate impact of noise using SVM in a binary classification
│   eval_noise_binary_xgb.py           <- Python script to evaluate impact of noise using XGBoost in a binary classification
│   eval_noise_mlabel_reg.py           <- Python script to evaluate impact of noise using Logistic Regression in a multilabel classification
│   eval_noise_mlabel_svm.py           <- Python script to evaluate impact of noise using SVM in a multilabel classification
│   eval_noise_mlabel_xgb.py           <- Python script to evaluate impact of noise using XGBoost in a multilabel classification
│
│── datasets                           <- all given in CSV format 
│       │── eval_features              <- generated datasets used within the evaluation protocol (adding features)
│       │── eval_noise                 <- generated datasets used within the evaluation protocol (adding noise)
│       └── main                       <- main train/test dataset used in main experiment
│
│── hextrato                           <- main library 
│       └── formulai.py                <- FormulAI Python core code
│
└── output                             <- output text files generated from each experiment
```
