# Install 

1. Create a new conda environment and activate it.

```bash
conda create -n sargent python=3.11 cmake=3.14.0
conda activate sargent

# Install PyTorch with CUDA support (choose ONE of these methods)
# Method 1: Using conda (recommended)
conda install pytorch=2.3.1 torchvision=0.18.1 torchaudio=2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Method 2: Using pip (alternative)
# pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements.txt
pip install git+https://github.com/microsoft/MoGe.git
```

2. Install [VGGT](https://github.com/facebookresearch/vggt) and its dependencies as a python package.

```
git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -e .
```

# Acknowledgements

This repository combines different models

* [VGGT](https://github.com/facebookresearch/vggt) under [dedicated VGGT License](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)
* [TEED](https://github.com/xavysp/TEED/tree/main) under [MIT License](https://github.com/xavysp/TEED/blob/main/LICENSE)
* [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) under [Apache License 2.0](https://github.com/google/mediapipe/blob/main/LICENSE)
* [MoGeV2](https://github.com/microsoft/MoGe) under [MIT License](https://github.com/microsoft/MoGe/blob/main/LICENSE)


