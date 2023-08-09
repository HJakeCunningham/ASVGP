# Actually Sparse Variational Gaussian Processes

![image](./images/sparse_matrices.png)

This repository includes the official implementation of [Actually Sparse Variational Gaussian processes](https://arxiv.org/abs/2304.05091), a sparse variational Gaussian process approximation, that utilises sparse linear algebra to efficiently scale low-dimensional Matern Gaussian processes to large numbers of datapoints.

Our implementation is built upon [GPFlow](https://github.com/GPflow/GPflow) and [banded_matrices](https://github.com/secondmind-labs/banded_matrices) packages. 

If you find this repository useful, please cite our paper
```bibtex
@inproceedings{cunningham2023actually,
  title={Actually Sparse Variational Gaussian Processes},
  author={Cunningham, Harry Jake and de Souza, Daniel Augusto and Takao, So and van der Wilk, Mark and Deisenroth, Marc Peter},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={10395--10408},
  year={2023},
  organization={PMLR}
}
```

## Installation

Our package requires installation of a development branch of [banded_matrices](https://github.com/secondmind-labs/banded_matrices) which is written in C++

1. Create fresh conda environement
```bash
conda create -n venv python=3.7
conda activate venv
```

2. Install tensorflow=2.4
```bash
pip install tensorflow==2.4
```

3. Clone `banded_matrices` package
```bash
git clone --branch awav/fix-banded-hashable-tensor https://github.com/secondmind-labs/banded_matrices.git
cd banded_matrices
```

4. Build python `banded_matrices` package (Note that his requires gcc version 7)
```bash
python setup.py sdist bdist_wheel
```

5. Install `banded_matrices` package
```bash
pip install dist/banded_matrices-0.0.6-*.whl
```

6. Install remaining requirements
```bash
pip install -r requirements.txt
pip install -e .
```

