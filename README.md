# FormulAI
Formulating controlled datasets to test AI systems.
Python package to generate synthetic hybrid datasets (including categorical and continuous features). 

## Project Organization

```
<files>
│   README.md                          <- Top-level README that overviews project
│   requirements.txt                   <- Required Python packages
│   generate_eval_dataset.py           <- Python script to generate the evaluation datasets
│   generate_main_dataset.py           <- Python script to generate the main dataset
│
<folders>
│── datasets                           <- all presented in both CSV and Python DataFrame objects formats (the latter saved with pickle)
│       │── eval                       <- small datasets used within the evaluation protocol 
│       └── main                       <- main datasets to be used in future experiments
│
└── 
```
