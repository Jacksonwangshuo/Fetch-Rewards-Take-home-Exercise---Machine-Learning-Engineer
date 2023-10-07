FROM python:3.8

ENV PATH=/root/.local/bin:$PATH

WORKDIR /home/homework/

COPY . .

RUN pip install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple pip 
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple flask
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas

CMD ["python", "app.py"]
