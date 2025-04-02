## Prepare the data

This repository comes without any data. You need to create a folder named "logfiles" and copy paste all the logfiles (csvs) into it, without any subfolder structure.

## Fit the models

It will create a new folder called ./model_fits. You may change the model space and parameter priors by tweaking the files contained in ./models.

## Select the model

Compare and select the model of interest for further analysis.

An "_augmented" dataset will be created with in the "compiled_data" subfolder. 

You may use it to get model-based fMRI regressors for ex.


## Setup the environment

In your terminal, type:
`conda env create -f jpndfmri-env-linux.yaml` or `conda env create -f jpndfmri-env-windows.yaml`