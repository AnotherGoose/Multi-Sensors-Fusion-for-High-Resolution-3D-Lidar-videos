clear all
close all



%Put in your own python path with the dependencies
pythonPath = 'C:\Users\jackm\anaconda3\envs\tfCPU\python.exe'
%If the python file still doesnt run, make sure the default application to
%open the python scripts points to your environment with dependencies
%installed

%Set python version
pyversion(pythonPath)

%Display Python Environment
pe = pyenv

%Run Python Script (Task-Based Adaptive Sampling)
system('TBAS.py')

%Run Matlab Scripts

