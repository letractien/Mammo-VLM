# Mammo-VLM
Project about mammography

# Activate enviroment
conda activate .venv-mammography

# Setup GPU
export CUDA_VISIBLE_DEVICES=1

# Move to folder
mv /root/.cache/kagglehub/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png/versions/1/* /root/letractien/Mammo-VLM/dataset/vindr/

# Install torch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html