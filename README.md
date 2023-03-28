# Protein Decoy
Codes of exploiting SparseConvNet to conduct protein decoy.
## Getting Started
### Setup 111
(1) The code is tested on Ubuntu 16.04 LTS & 18.04 LTS with PyTorch 1.3 CUDA 10.1 installed.
```shell
conda install pytorch==1.3.0 cudatoolkit=10.1 -c pytorch
```
(2) Compile SparseConvNet ops.
```shell
cd lib/
python setup.py develop
```
(3) Install required libraries.
```shell
cd lib/
pip install -r requirement.txt
```

### Data Preparation
(1) To transfer original PDB files to XYZ files, you need prepare [LIG_TOOL](https://github.com/realbigws/PDB_Tool).
```shell
git clone https://github.com/realbigws/PDB_Tool.git
```
(2) Modify the path in `datasets/prepare_pdb_to_xyz.py` and run
```shell
cd datasets/
python prepare_pdb_to_xyz.py
```

### Training and Evaluation
Before the training phase, you can modify your model setting in `cfgs/*yaml`, and choose the `yaml` file you want to use.
```shell
python train.py --log_dir [EXP_NAME] --cfg_file [YOUR_CFG_FILE] --gpu [GPU_ID]
```
For instance, if you want to nane the experiment log with `SparseConv_default` and use `cfgs/SparseConv-Cath-Decoys-20201231.yaml`, you can run with
```shell
python train.py --log_dir SparseConv_default --cfg_file cfgs/SparseConv-Cath-Decoys-20201231.yaml --gpu 0
```
In the first-time training, it will cost time to collect data.

Currently, we support multiple GPU training:
```shell
python -m torch.distributed.launch --nproc_per_node=[GPU_ID] train_dist.py --cfg_file cfgs/SparseConv-Cath-Decoys-20201231.yaml --gpu 1 2
```

### Test 
```shell
python test.py --log_dir [EXP_NAME] --data_path [DATA_PATH]
```
The output will be shown in `EXP_NAME/prediction`. There are some additional argparse you may need:
* `filename_suffix`: the suffix name of data
* `eval`: evaluate the prediction results
* `residue_level`: output the residue level results
* `show`: visualize the errors, and save in `EXP_NAME/visualization`
* `vote_num`: voting number

Visualization results can be open with `MeshLab`.
![](figures/results.png)

### Test using pretrained parameters
```shell
cd test/
```
Download pretrained parameters from [here](https://cuhko365-my.sharepoint.com/:f:/g/personal/220019151_link_cuhk_edu_cn/EgZHv6wR9RpOkVtjyPPHS3oB-aYgawSwULZZctDfvqSyFQ?e=HKbsdz)
, then copy parameter files `ckpt_decoy8k.pth` and `ckpt_cath2084.pth` to `./checkpoint/`

pdb files should be preprocessed into `.xyz` files and `.pkl` files first. Notice `LIG_Tool` should be compiled.
```shell
python preprocess.py --LIG_path Your_LIG_Path/util/PDB_To_XYZ --pdbpath [PDB_FILE_PATH] --xyzpath [XYZ_FILE_PATH] --pklpath [PKL_FILE_PATH] --workers 4
```
run test.py to make prediction:
```shell
python test.py --gpu 0 --data_path [PKL_FILE_PATH] --xyz_path [XYZ_FILE_PATH] --log_dir [SAVE_PATH] --ckpt_path ./checkpoint/ckpt_decoy8k.pth --cfg_path ./checkpoint/SparseConv_AtomR.yaml --vote_num 3
```
A demo can be runned through:
```shell
python preprocess.py --LIG_path Your_LIG_Path/util/PDB_To_XYZ
python test.py --gpu 0
```
After running the code above, atom-level, residue-level, golbel-level lddt can all be predicted and saved in `.npy` files, an example of reading these results is:
```shell
import numpy as np
results=np.load('examples/prediction/all_lddt/1d2zB_nat12.npy',allow_pickle=True)
atom_lddt=results.item().get('atom_lddt')
residue_lddt=results.item().get('residue_lddt')
global_lddt=results.item().get('global_lddt')
```
