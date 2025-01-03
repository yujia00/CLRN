# Multi-behavior aware recommendation with joint contrastive learning and reinforced negative sampling
## Introduction
This is the Pytorch implementation for our paper. In this work, we propose a multi-behavior aware recommendation model, CLRN, which integrates contrastive learning and reinforced negative sampling. It aims to address the shortcomings of existing methods in modeling behavior dependency relationships, handling skewed data distributions, and optimizing the quality of negative samples.

## Dataset
We use three processed datasets: IJCAI, Tmall, Retail

|              | IJCAI  | Tmall | Retail |
|--------------|----------|----------|---------|
| #Users       | 17,435  | 31,882   | 2,174 |
| #Items       | 35,920   | 31,323   | 30,113  |
| #Interactions| 799,368  | 1,451,219| 97,381|
| #Target Interaction | 131,685  | 167,862  | 9,551 |

## Enviroment
The codes of CLRN are implemented and tested under the following development environment:
- Python 3.12.2
- torch==2.2.2+cu121
- scipy==1.12.0
- tqdm==4.65.0

## Model Structure
![CLRN model](https://github.com/user-attachments/assets/4a24e476-9ed0-4589-b465-da15a2eeda0b)
