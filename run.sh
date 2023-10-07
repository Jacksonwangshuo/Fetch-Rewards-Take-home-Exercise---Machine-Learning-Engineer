#!/bin/bash

#### 环境依赖 ####
pip install flask
pip install numpy
pip install torch
pip install pandas
#################

#### 启动训练 ####
python train.py
#################

#### 启动 web 服务在线预测 ####
python app.py
############################

#### 将项目打包成 docker ####
cd project
docker build -t project_name:v1 .
docker run -it -d -p 8888:8888 --restart=always project_name:v1
###########################