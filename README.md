# pflda
The numerical experiment code of the paper Personalized Federated Learning via Domain Adaptation with an Application to Distributed 3D Printing.

We propose PDA, a personalized federated learning approach built on domain adaptation, in which clients borrow strength from each other while each keeps a model tailored to its own data.

## 1. Sine regression 
sine_regression folder contains all the code needed for reproducing results in section 4.1

for instance, try

python3 sineregression.py --seed=6  --fed='PDA'  

The fed argument specifies which algorithm to use, three options are 'PDA', 'ditto', and 'indiv'. 

All results will be generated to sine_outputs folder

## 2. Case study on image classification
case_study folder contains all the code needed for reproducing results in section 4.2. Some classes are inherited from the [Domainbed repository](https://github.com/facebookresearch/DomainBed).


## 3. 3D printing
printer_dataset folder contains all the code needed for reproducing results in section 4.3

## Citation

[Paper](https://doi.org/10.1080/00401706.2022.2157882) (Technometrics, 2023). If you use this code, please cite:

```bibtex
@article{shi2023personalized,
  title={Personalized Federated Learning via Domain Adaptation with an Application to Distributed {3D} Printing},
  author={Shi, Naichen and Al Kontar, Raed},
  journal={Technometrics},
  volume={65},
  number={3},
  pages={328--339},
  year={2023},
  doi={10.1080/00401706.2022.2157882}
}
```
