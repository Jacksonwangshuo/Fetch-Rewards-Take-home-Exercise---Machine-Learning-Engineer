#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
df = pd.read_csv('data_daily_2022.csv')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    data_dict = json.loads(request.get_data())
    month = data_dict['month']
    
    count = 0
    for idx, row in df.iterrows():
        date = str(row['# Date'])
        receipt_count = int(row['Receipt_Count'])
        if month in date:
            count += receipt_count
    return_dict = {"count": count, "error": ""}
    return jsonify(return_dict)


if __name__ == '__main__':
    app.run(debug=True, port=8888, host='0.0.0.0')
