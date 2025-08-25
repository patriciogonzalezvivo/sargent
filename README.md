

# Install 

1. Create a new conda environment and activate it.

```
conda create -n sargent python=3.11 cmake=3.14.0
conda activate sargent
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

2. Install [VGGT](https://github.com/facebookresearch/vggt) and its dependencies as a python package.

```
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```



