# Mammo-VLM
Project about mammography

# Activate enviroment
conda activate .venv-mammography

# Deactivate enviroment
conda deactivate

# Setup GPU
export CUDA_VISIBLE_DEVICES=0

# Move to folder
mv /root/.cache/kagglehub/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png/versions/1/* /root/letractien/Mammo-VLM/dataset/vindr/

# Install torch
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

# Predict from Qwen
/root/letractien/Mammo-VLM/out/detect_qwen/0f0551f4edb5494b0d8765c23fe421ae/a37e508fc994c1c7a846ec23edfb400f_bbox.png

# Video demo
[![Video demo](https://img.youtube.com/vi/h88MwGj0T-U/0.jpg)](https://www.youtube.com/watch?v=h88MwGj0T-U)
