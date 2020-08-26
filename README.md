# Optimization of Graph Neural Networks with Natural Gradient Descent

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optimization-of-graph-neural-networks-with/node-classification-on-citeseer)](https://paperswithcode.com/sota/node-classification-on-citeseer?p=optimization-of-graph-neural-networks-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optimization-of-graph-neural-networks-with/node-classification-on-cora)](https://paperswithcode.com/sota/node-classification-on-cora?p=optimization-of-graph-neural-networks-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optimization-of-graph-neural-networks-with/node-classification-on-pubmed)](https://paperswithcode.com/sota/node-classification-on-pubmed?p=optimization-of-graph-neural-networks-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optimization-of-graph-neural-networks-with/node-classification-on-citeseer-with-public)](https://paperswithcode.com/sota/node-classification-on-citeseer-with-public?p=optimization-of-graph-neural-networks-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optimization-of-graph-neural-networks-with/node-classification-on-pubmed-with-public)](https://paperswithcode.com/sota/node-classification-on-pubmed-with-public?p=optimization-of-graph-neural-networks-with)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/optimization-of-graph-neural-networks-with/node-classification-on-cora-with-public-split)](https://paperswithcode.com/sota/node-classification-on-cora-with-public-split?p=optimization-of-graph-neural-networks-with)

This repository contains the implementaion of the [Optimization of Graph Neural Networks with Natural Gradient Descent](https://arxiv.org/abs/2008.09624). Most of the code is adapted from [github.com/rusty1s/pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) and [github.com/Thrandis/EKFAC-pytorch](https://github.com/Thrandis/EKFAC-pytorch). To duplicate the results reported in the paper, follow the subsequent steps in order. 

- Clone the repository and change your current directory:
```
git clone https://github.com/russellizadi/ssp
cd ssp
```
- Create a new `conda` environment using the default `environment.yml`:
```
conda env create
```
- Activate the default environment:
```
conda activate ssp
```
- Go to the `experiments` folder:
```
cd experiments
```
- Run all the experiments performed in the paper:
```
./run.sh
```
