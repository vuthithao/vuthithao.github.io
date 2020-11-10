---
layout: post
title: Docker Guide
---

# Set up môi trường cho máy GPU
Chạy `nvidia-smi`, nếu có 1 bảng thông số hiện ra nghĩa là máy đã được cài driver cho GPU, có thể bỏ qua mục này.

- Nếu chưa có, cài driver cho GPU theo các bước sau:
```bash
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-driver-418
```
- Khởi động máy `sudo reboot`
- Kiểm tra lại lệnh `nvidia-smi`
- Nếu có bảng hiện ra là đã cài đặt thành công

# Cài đặt nvidia-docker
- Add repository
```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
```
- Cài đặt nvidia-docker2
```bash
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
```
- Test kết quả
```bash
docker run --runtime=nvidia --rm nvidia/cuda:10.0-base nvidia-smi
```

# Build, run docker

## Clone example project từ gitlab
```bash
git clone http://gitlab.giaingay.io/vuthithao/ken.git
cd ken
```

```
├── model
│   ├── Randomforest.obj
│   └── scaler.sav
├── api.py
├── build_L3DB.py
├── classifier.py
├── post_request.py
├── requirements.txt
└── updateDB.py
``` 
- Đây là cấu trúc một project đơn giản để phân loại sử dụng Randomforest, cung cấp service phân loại bằng cách chạy
```bash
python api.py
```
## Đóng gói docker

### Docker CPU

##### Bước 1: Tạo `Dockerfile`
###### Ví dụ:
```
# Use an official Python runtime as a parent image
FROM python:3.6.5
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y python3-pip python3-dev && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python3", "api.py"]
```

- Nếu sử dụng GPU uncomment dòng `#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04`
- Bắt buộc phải có file requirements.txt (bao gồm các module cần cài đặt cho project)
- CMD ["python3", "api.py"] có nghĩa docker chạy dòng lệnh `python3 api.py`

##### Bước 2: Build image
```bash
docker build -t <image_name>:<version> .
```
###### Ví dụ:
```bash
docker build -t image_name:latest .
```

##### Bước 3: Run image

```bash
docker run --name <container name> -d -p <local-port>:<api-port> image_name:latest
```
- <local-port>: port đầu ra thực tế trên server chạy docker
- <api-port>: port đầu ra trong code (được định nghĩa trong `api.py`)

###### Ví dụ:
```bash
docker run --name ken -d -p 3000:4000 image_name:latest
```

##### Bước 4: Kiểm tra kết quả
```bash
docker container ls -a
```
- Nếu container của bạn đang chạy, bạn đã thành công, có thể sử dụng `Postman` để kiểm tra API của bạn như thông thường.
- Check logs của container sử dụng
```bash
docker container logs ken
```

### Docker GPU
- Nếu sử dụng GPU uncomment dòng `#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04`
- Thay thế `docker` bằng `nvidia-docker`




