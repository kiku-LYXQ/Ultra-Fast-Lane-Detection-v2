export DALI_DISABLE_NVML=1   (for wsl2, dali should disable NVML)

sudo apt install openssh-server

sudo systemctl start ssh

sudo systemctl enable ssh

scp \\wsl.localhost\Ubuntu-24.04\home\lxy\lxy_project\Ultra-Fast-Lane-Detection-v2\weights\culane_res34.onnx lxy@172.23.241.211:/home/lxy/Desktop/lxy_project/Ultra-Fast-Lane-Detection-v2/weights