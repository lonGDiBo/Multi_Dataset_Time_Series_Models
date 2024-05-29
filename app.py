from flask import Flask
from flask import *
from markupsafe import escape
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
from sklearn.preprocessing import MinMaxScaler



app = Flask(__name__) # create an app instance



from flask import abort, redirect, url_for


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)

def split_data(data, split_ratio=0.8):
    train_size = int(len(data) * split_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test 
    
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
            if not stock_name:
                  return render_template('index.html', message="Please choose a dataset"), 400 
            file_path = f"{stock_name}.csv"
            data = load_data(f"./dataset/{file_path}")
            global_data = data.copy()
            global_name = stock_name
            count_data = data.shape[0]
            data_loaded = True   
            columns = data.columns.tolist()
            num_columns = len(columns)
    return render_template('index.html', data=data.to_html() if data is not None else None, data_loaded=data_loaded,columns = columns,stock_name=stock_name,
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
                plt.xlabel('Date') 
            elif global_name == 'Weather_WS':
                data_1['Date Time'] = pd.to_datetime(data_1['Date Time'])
                plt.plot(data_1['Date Time'],data_1[column_name])
                plt.xlabel('Date') 
            elif global_name == 'weather-HCM':
                data_1['date'] = pd.to_datetime(data_1['date'])
                plt.plot(data_1['date'],data_1[column_name])
                plt.xlabel('Date')  # Add x-axis label

            plt.title(column_name)
            plt.savefig('static/images/plot.png')
    return jsonify(column_name=column_name)

@app.route('/model', methods=['GET', 'POST'])
def Predict():
    global global_data 
    global global_name
    if global_name == 'GOOGLE' or global_name == 'APPLE' or global_name == 'AMAZON':
        data_without_time = global_data.drop(['Date','Volume'], axis=1)
        train, test = split_data(scale_data(data_without_time.values.reshape(-1,1)))      
    elif global_name == 'Weather_WS':
        data_without_time = global_data.drop('Date Time', axis=1)
        train, test = split_data(scale_data(data_without_time.values.reshape(-1,1)))
    elif global_name == 'weather-HCM':
        data_without_time = global_data.drop(['date','wind_d'], axis=1)
        train, test = split_data(scale_data(data_without_time.values.reshape(-1,1)))
    
    train_length = train.shape[0]
    test_length = len(test)
    
    return jsonify(train=train.tolist(), test=test.tolist(), train_length=train_length, test_length=test_length)