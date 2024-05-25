from flask import Flask
from flask import *
from markupsafe import escape
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)



from flask import abort, redirect, url_for


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

    
@app.route('/')
def index():
    return render_template('login.html')

global_data = None
global_name = None

@app.route('/data', methods=['GET', 'POST'])
def data():
    data_loaded = False
    stock_name = None
    data = None
    columns = None
    num_columns = None
    count_data = None
    file_path= None
    
    global global_data
    global global_name
    if request.method == 'POST':
        if 'data' in request.form:
            stock_name = request.form['data']
            file_path = f"{stock_name}.csv"
            data = load_data(f"./dataset/{file_path}")
            global_data = data.copy()
            global_name = stock_name
            count_data = data.shape[0]
            data_loaded = True   
            columns = data.columns.tolist()
            num_columns = len(columns)
    return render_template('main.html', data=data.to_html() if data is not None else None, data_loaded=data_loaded,columns = columns,stock_name=stock_name,
                           count_data=count_data,num_columns=num_columns,file_path=file_path)

@app.route('/eda_column', methods=['GET', 'POST'])
def eda_data():
    global global_data  
    global global_name
    data_1 = global_data.copy()
    column_name = None
    if request.method == 'POST':
        if 'column' in request.form:
            column_name = request.form.get('column')
            plt.figure(figsize=(10, 5))
            if global_name == 'GOOGLE' or global_name == 'APPLE' or global_name == 'AMAZON':
                data_1['Date'] = pd.to_datetime(data_1['Date'])
                plt.plot(data_1['Date'],data_1[column_name])
            elif global_name == 'Weather_WS':
                data_1['Date Time'] = pd.to_datetime(data_1['Date Time'])
                plt.plot(data_1['Date Time'],data_1[column_name])
            elif global_name == 'weather-HCM':
                data_1['date'] = pd.to_datetime(data_1['date'])
                plt.plot(data_1['date'],data_1[column_name])
            plt.title(column_name)
            plt.savefig('static/images/plot.png')
    return jsonify(column_name=column_name)