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

3. Clone `banded_matrices` package
```bash
git clone https://github.com/secondmind-labs/banded_matrices.git
cd banded_matrices
```

4. Switch branch to `awav/fix-banded-hashable-tensor`
```bash
git fetch
git checkout -b origin/awav/fix-banded-hashable-tensor 
```

5. Build python `banded_matrices` package (Note that his requires gcc version 7)
```bash
python setup.py sdist bdist_wheel
```

6. Install `banded_matrices` package
```bash
pip install dist/banded_matrices-0.0.7-*.whl
```