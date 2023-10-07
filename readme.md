# **Fetch Rewards Take-home Exercise - Machine Learning Engineer**

### Files

----

- train.py
- app.py
- Dockerfile
- static/
- templates/

### How to run

#### In Python

1. Run the python script

```shell
python app.py
```

2. Access http://127.0.0.1:8888

#### In Docker

1. Build the image

```shell
cd project
docker build -t project_name:v1 .
```

2. Run the container

```shell
docker run -it -d -p 8888:8888 --restart=always project_name:v1
```

3. Access http://127.0.0.1:8888
