from flask import Flask
from flask import *
import numpy as np
import pandas as pd
from markupsafe import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime as dt
from keras.models import Sequential 
from keras.layers import Dense, LSTM 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
from sklearn.metrics import mean_squared_error


app = Flask(__name__) # create an app instance



from flask import abort, redirect, url_for


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna()
    return data

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data)

def split_data_default(data, split_ratio=0.8):
    train_size = int(len(data) * split_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test 

def split_data_new(data, split_ratio):
    train_size = int(len(data) * split_ratio)
    train, test = data[:train_size], data[train_size:]
    return train, test

def to_sequences(dataset,timestep , seq_size=1): 
    x = []
    y = []
    for i in range(0,len(dataset)-seq_size-1,timestep):
        window = dataset[i:(i+seq_size), 0]
        x.append(window)
        y.append(dataset[i+seq_size, 0])
    return np.array(x),np.array(y)

def to_sequences_multivariate_varnn(dataset,p):
    x = []
    y = []
    for i in range(p, len(dataset)):
        x.append(dataset[i - p:i, 0:dataset.shape[1]])
        y.append(dataset[i:i + 1, 0:dataset.shape[1]])
    x = np.array(x)
    y = np.array(y)
    return x,y.reshape(y.shape[0], y.shape[2])

def to_sequences_multivariate_lstm(dataset,p):
    x = []
    y = []
    for i in range(p, len(dataset)):
        x.append(dataset[i - p:i, 0:dataset.shape[1]])
        y.append(dataset[i:i + 1, 0:dataset.shape[1]])
    x = np.array(x)
    y = np.array(y)
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2]),y.reshape(y.shape[0], y.shape[2])

def model_ffnn_exist(seq_size, hidden_neurons, weights_file):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=seq_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model.load_weights(weights_file)
    return model

def model_ffnn_new(train, test, hidden_layers, seq_size, hidden_neurons, epoch, batchsize):
    trainX, trainY = to_sequences(train,1, seq_size)
    testX, testY = to_sequences(test,1, seq_size)
    model = Sequential()
    for j in range(1, hidden_layers+1):
        model.add(Dense(hidden_neurons, input_dim=seq_size, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model.fit(trainX, trainY, validation_data=(testX, testY), verbose=0, epochs=epoch, batch_size=batchsize)
    return model

def get_param_ffnn(default_split_ratio,default_hidden_neurons,default_seq_size,default_epochs,default_batch_size,default_hidden_layers):
   split_ratio_get = global_parameters.get('splitdata',default_split_ratio)
   if split_ratio_get == 'split73':
       split_ratio = 0.7
   else:
       split_ratio = 0.8
   hidden_neurons = int(global_parameters.get('Hidden_Neurons', default_hidden_neurons))
   seq_size = int(global_parameters.get('Data_window_size', default_seq_size))
   epochs = int(global_parameters.get('Epoch', default_epochs))
   batch_sizes = int(global_parameters.get('Batch_size', default_batch_size))
   hidden_layers = int(global_parameters.get('Hidden_Layers', default_hidden_layers))
   return split_ratio, hidden_neurons,seq_size,epochs,batch_sizes,hidden_layers


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
    filename = None
    
    global global_data
    global global_name
    if request.method == 'POST':
        if 'data' in request.form and request.form['data']:  # Check if the request has the data part
            stock_name = request.form['data']
            file_path = f"./dataset/{stock_name}.csv"
            data = load_data(file_path)
            global_data = data.copy()
            global_name = stock_name
            count_data = data.shape[0]
            data_loaded = True   
            columns = data.columns.tolist()
            num_columns = len(columns)
        elif 'text' in request.files and request.files['text'].filename != '':  # Check if the request has the file part
            file = request.files['text']
            filename = secure_filename(file.filename)
            file_path = os.path.join('./dataset/new_dataset/', filename)
            file.save(file_path)  # Save the file to a destination
            data = load_data(file_path)
            global_data = data.copy()
            filename_without_extension = filename.split(".")[0]
            global_name = filename_without_extension
            count_data = data.shape[0]
            data_loaded = True   
            columns = data.columns.tolist()
            num_columns = len(columns)
        else:
            return render_template('index.html', message="Please choose a dataset or upload a file"), 400
        
    return render_template('index.html', data=data.to_html() if data is not None else None, data_loaded=data_loaded,columns = columns,stock_name=stock_name,
                           count_data=count_data,num_columns=num_columns,file_path=file_path, filename=filename, global_name=global_name)

   
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
            else:
                plt.plot(data_1[column_name])
            plt.title(column_name)
            plt.savefig('static/images/plot.png')
    return jsonify(column_name=column_name)



@app.route('/model', methods=['GET', 'POST'])
def Predict():
    global global_data 
    global global_name
    algorithm =None
    column_prediction = None
    useExistingModel = None
    train = []
    test = []
    testScore_mse = 0
    model_path_ffnn = 'Model/Apple/FFNN/'
    algorithm = request.form.get('algorithm')
    column_prediction = request.form.get('column_prediction')
    useExistingModel = request.form.get('useExistingModel') 
    model = None
            
    if  algorithm == 'algorithm-ffnn':        
        if global_name == 'GOOGLE':
            train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))               
        elif global_name == 'APPLE':
            train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))
            if column_prediction == 'Open': 
                # Default parameters
                default_hidden_layers = 1
                default_hidden_neurons = 16
                default_seq_size = 18
                default_epochs = 400
                default_batch_size = 32
                default_split_ratio = 0.8
                # Get parameters
                split_ratio,hidden_neurons,seq_size,epochs,batch_sizes,hidden_layers = get_param_ffnn(default_split_ratio,default_hidden_neurons,default_seq_size,default_epochs,default_batch_size,default_hidden_layers)             
                # Check if the user wants to use an existing model
                if useExistingModel == 'on':
                    model_path = model_path_ffnn + 'FFNN_Model_Apple_Open.h5'
                    model = model_ffnn_exist(default_seq_size, default_hidden_neurons, model_path)
                    x,y = to_sequences(test,1,18)
                    test_pred = model.predict(x)
                    testScore_mse = mean_squared_error(y, test_pred)
                    train_length = train.shape[0]
                    test_length = len(test)
                    plt.clf() 
                    plt.plot(y,label="Actual value")
                    plt.plot(test_pred,label="Predicted value")
                    plt.legend()
                    plt.savefig('static/images/plot_predict.png')
                    return jsonify(algorithm=algorithm, column_prediction=column_prediction, train_length=train_length, test_length=test_length, testScore_mse=testScore_mse, hidden_neurons=default_hidden_neurons, seq_size=default_seq_size, hidden_layers=default_hidden_layers, epochs=default_epochs, batch_sizes=default_batch_size, split_ratio=default_split_ratio)                                      
                else: # Train a new model
                    if split_ratio == 0.7:
                        train_new, test_new = split_data_new(scale_data(global_data[column_prediction].values.reshape(-1,1)),split_ratio)
                        model = model_ffnn_new(train_new, test_new,hidden_layers, seq_size, hidden_neurons, epochs, batch_sizes)
                        x,y = to_sequences(test_new,1,seq_size)      
                    else: 
                        model = model_ffnn_new(train, test,hidden_layers, seq_size, hidden_neurons, epochs, batch_sizes)
                        x,y = to_sequences(test,1,seq_size) 
                    test_pred = model.predict(x)
                    testScore_mse = mean_squared_error(y, test_pred)
                    plt.clf() 
                    plt.plot(y,label="Actual value")
                    plt.plot(test_pred,label="Predicted value")
                    plt.legend()
                    plt.savefig('static/images/plot_predict.png')
                    return jsonify(algorithm=algorithm, column_prediction=column_prediction, testScore_mse=testScore_mse, hidden_neurons=hidden_neurons, seq_size=seq_size, hidden_layers=hidden_layers, epochs=epochs, batch_sizes=batch_sizes, split_ratio=split_ratio)  
        elif global_name == 'AMAZON':
            train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))
        elif global_name == 'Weather_WS':
            train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))    
        elif global_name == 'weather-HCM':
            train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))    
        else:
            train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))       
    else:
        return render_template('index.html', message="Please choose a model"), 400

    
global_parameters = {}

@app.route('/save_parameters', methods=['GET', 'POST'])
def save_param():
    global global_parameters
    parameters = request.get_json()
    for param_name, value in parameters.items():
        print(f"Parameter {param_name} has value {value}")
    global_parameters = parameters

    name = global_parameters.get('name', None)
    if name is not None:
        print(f"The value of 'name' is {name}")
    return jsonify({'message': 'Parameters received'}), 200
    
    
if __name__ == '__main__':
    app.run(debug=True) 