# Relational Networks in PyTorch

A PyTorch implementation of Relational Networks by Santoro et al (https://arxiv.org/abs/1706.01427)

	Santoro, A., Raposo, D., Barrett, D. G., Malinowski, M., Pascanu, R., Battaglia, P., & Lillicrap, T. (2017). A simple neural network module for relational reasoning. In Advances in neural information processing systems (pp. 4967-4976).

For evaluation purposes we use an existing version of the Sort-Of-CLEVR dataset, available through kaggle. This dataset contains of 10000 images with questions encoded in vectors.

## Installation

After cloneing this repository you can either use your own pytorch environment or use the following simple setup script for a new environment:

```
$ make create_environment
$ source activate pytorch_p27
$ make requirements
```

## Usage

You can directly use the `src/main.py` file and the commands specified there or start the scripts with `make train` and `make train_mlp` for the two network variants.

## Evaluation

