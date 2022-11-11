# pflda
The numerical experiment code of the paper Personalized Federated Learning via Domain Adaptation with an Application to Distributed 3D Printing.

## 1. Sine regression 
sine_regression folder contains all the code needed for reproducing results in section 4.1

for instance, try

python3 sineregression.py --seed=6  --fed='PDA'  

The fed argument specifies which algorithm to use, three options are 'PDA', 'ditto', and 'indiv'. 

All results will be generated to sine_outputs folder

## 2. Case study on image classification
case_study folder contains all the code needed for reproducing results in section 4.2. Some classes are inherited from the (Domainbed repository)[https://github.com/facebookresearch/DomainBed].


## 3. 3D printing
printer_dataset folder contains all the code needed for reproducing results in section 4.3
