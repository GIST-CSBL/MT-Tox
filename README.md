# MT_Tox

## Information
Official github repository of the MT-Tox

Contact Info:

15pms@gm.gist.ac.kr

hjnam@gist.ac.kr

<br />

## Environment setting (Anaconda)

You have to download pytorch and dgl wheel files.

Download dgl-1.1.1+cu113-cp37-cp37m-manylinux1_x86_64.whl file from [dgl](https://data.dgl.ai/wheels/cu113/repo.html)

Download torch-1.12.1+cu113-cp37-cp37m-linux_x86_64.whl file from [pytorch](https://download.pytorch.org/whl/torch/)

```
conda create -n MT_Tox python=3.7.16

conda activate MT_Tox

pip install -r requirements.txt

pip install torch-1.12.1+cu113-cp37-cp37m-linux_x86_64.whl

pip install dgl-1.1.1+cu113-cp37-cp37m-manylinux1_x86_64.whl
```

After setting conda environment, use the following command to use the jupyter notebook
> python -m ipykernel --user --name MT_Tox

<br />

## MT_Tox file explanation
- data : Folder that contains the pre-training & fine-tuning datasets for experiments
- model : Folder that contains the `.py` files and pre-trained model weights for model training
- Tox21_multitask_training.ipynb : Jupyter notebook files for implementing Tox21 (*in vitro* toxicological information) trainig
- in_vivo_finetuning.ipynb : Jupyuter notebook files for implementing *in vivo* toxicity fine tuning
- in_vivo_inference.ipynb : Jupyter notebook files for *in vivo* toxicity inference for any molecular dataset 

