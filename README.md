# E2E-FS
E2E-FS: An End-to-End Feature Selection Method for Neural Networks

## Setup Instructions

### The python environment is included in the file requirements.txt.
Run the command:
 
    conda create --name e2efs --file ./requirements.txt

### Instructions for compiling the extern liblinear library:
Run the following commands:

    cd extern/liblinear
    make all 
    cd python
    make lib 

## Running the code
Activate the environment:

    conda activate e2efs 

All scripts are included into the script folder. To run it:

    PYTHONPATH=.:$PYTHONPATH python scripts/microarray/run_all.py
    PYTHONPATH=.:$PYTHONPATH python scripts/fs_challenge/run_all.py
    PYTHONPATH=.:$PYTHONPATH python scripts/deep/run_all.py
