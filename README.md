# E2E-FS
E2E-FS: An End-to-End Feature Selection Method for Neural Networks

## The python environment is included in the file requirements.txt.
Run the command:
 
    conda create --name e2efs --file ./requirements.txt

## Instructions for compiling the extern liblinear library:
Run the following commands:

    cd extern/liblinear
    make all 
    cd python
    make lib 
