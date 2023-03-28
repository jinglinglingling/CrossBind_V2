### CrossBind
Codes of protein residue binding prediction.
## Getting Started
### Setup
(1) Pytorch version.
```shell
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
```
(2) Compile SparseConvNet ops.
```shell
cd lib/
python setup.py develop


### Data Preparation
(1) You can download the dataset (DNA/RNA pdb files) (http://www.csbio.sjtu.edu.cn/bioinf/GraphBind/ , https://github.com/biomed-AI/GraphSite)
```
(2) To transfer original PDB files to XYZ files, you need prepare [LIG_TOOL](https://github.com/realbigws/PDB_Tool).
```shell
git clone https://github.com/realbigws/PDB_Tool.git
```
(3) Modify the path in `datasets/prepare_pdb_to_xyz.py` and run
```shell
cd datasets/
python prepare_pdb_to_xyz.py
```

### Training and Evaluation
Before the training phase, you can modify your model setting in `cfgs/*yaml`, and choose the `yaml` file you want to use.
```
Run full version of CrossBind:
python CrossBind.py --log_dir SparseConv_default --cfg_file cfgs/SparseConv-Cath-Decoys-Clf-Only.yaml --gpu 0

```
In the first-time training, it will cost time to collect data.

