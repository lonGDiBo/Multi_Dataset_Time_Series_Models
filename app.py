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
from sklearn.metrics import mean_squared_error,mean_absolute_error
import time

app = Flask(__name__) # create an app instance



from flask import abort, redirect, url_for


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def load_data_new(file_path):
    data = pd.read_csv(file_path)
    df_numerical = data.select_dtypes(include=[float, int])
    df_final= df_numerical.dropna(axis=1)
    return df_final

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def scale_data_original(test,predictions,scaler):
    predictions_actual = scaler.inverse_transform(predictions.reshape(1, -1))
    test_actual = scaler.inverse_transform(test.reshape(1, -1))
    return predictions_actual, test_actual

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
    for i in range(0,len(dataset)-seq_size,timestep):
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

# ----------------------------------START FFNN ----------------------------------
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

def get_param_ffnn_datasetNew():
   split_ratio_get = global_parameters.get('splitdata')
   if split_ratio_get == 'split73':
       split_ratio = 0.7
   else:
       split_ratio = 0.8
   hidden_neurons = int(global_parameters.get('Hidden_Neurons'))
   seq_size = int(global_parameters.get('Data_window_size'))
   epochs = int(global_parameters.get('Epoch'))
   batch_sizes = int(global_parameters.get('Batch_size'))
   hidden_layers = int(global_parameters.get('Hidden_Layers'))
   return split_ratio, hidden_neurons,seq_size,epochs,batch_sizes,hidden_layers
# ---------------------------------- END FFNN ----------------------------------
# ---------------------------------- LSTM ----------------------------------
def LSTM_exist(train,outputs,seq,hidden_neural,weights_file):
    seq_size = seq
    trainX_LSTM, trainY_LSTM = to_sequences_multivariate_lstm(train,seq_size)
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(hidden_neural, return_sequences=False, input_shape= (trainX_LSTM.shape[1], 1)))
    model_LSTM.add(Dense(outputs))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model_LSTM.load_weights(weights_file)
    return model_LSTM

def LSTM_new( train,test,outputs,seq,hidden_neural,epochs,batch_size,hidden_layers):
    seq_size = seq
    trainX_LSTM, trainY_LSTM = to_sequences_multivariate_lstm(train,seq_size)
    testX_LSTM, testY_LSTM = to_sequences_multivariate_lstm(test,seq_size)
    model_LSTM = Sequential()
    for j in range(1, hidden_layers+1):
         model_LSTM.add(LSTM(hidden_neural, return_sequences=False, input_shape= (trainX_LSTM.shape[1], 1)))
    model_LSTM.add(Dense(outputs))
    model_LSTM.compile(loss='mean_squared_error', optimizer='adam', metrics = ['acc'])
    model_LSTM.fit(trainX_LSTM, trainY_LSTM, validation_data=(testX_LSTM, testY_LSTM), verbose=0, epochs=epochs, batch_size=batch_size)
    return model_LSTM

def LSTM_Predict(result_LSTM,testY_LSTM,predict_LSTM_real,textY_LSTM_real,arrayValue,column_prediction):
    predict_LSTM = result_LSTM[:,arrayValue.index(column_prediction)]
    textY_LSTM = testY_LSTM[:,arrayValue.index(column_prediction)]  
    testScore_mse, testScore_rmse, testScore_mae = calculate_metrics(textY_LSTM, predict_LSTM)                     
    
    predict_LSTM_real = predict_LSTM_real[:,arrayValue.index(column_prediction)]
    textY_LSTM_real = textY_LSTM_real[:,arrayValue.index(column_prediction)]                   
    testScore_mse_real, testScore_rmse_real, testScore_mae_real = calculate_metrics(textY_LSTM_real, predict_LSTM_real)                      
    eda_model_LSTM(textY_LSTM,predict_LSTM,column_prediction)
    return testScore_mse, testScore_rmse, testScore_mae, testScore_mse_real, testScore_rmse_real, testScore_mae_real


def calculate_metrics(test, predictions):
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    return mse, rmse, mae



def eda_model_FFNN(y,test_pred,column_prediction):
    plt.clf() 
    plt.plot(y,label="Actual value")
    plt.plot(test_pred,label="Predicted value")
    plt.title('FFNN Predictions of {} Values Compared to Actuals'.format(column_prediction))
    plt.legend()
    plt.savefig('static/images/plot_predict.png')

def eda_model_LSTM(y,predict_LSTM_Open,column_prediction):
    plt.clf() 
    plt.plot(y,label="Actual value")
    plt.plot(predict_LSTM_Open,label="Predicted value")
    plt.title('LSTM Predictions of {} Values Compared to Actuals'.format(column_prediction))
    plt.legend()
    plt.savefig('static/images/plot_predict.png')
    

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
    data_out = None
    filename = None
    output_columns = None
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
            
            data_out = load_data_new(file_path)
            output_columns = data_out.columns.tolist()   
            
            columns = data.columns.tolist()
            num_columns = len(columns)
        else:
            return render_template('index.html', message="Please choose a dataset or upload a file"), 400
        
    return render_template('index.html', data=data.to_html() if data is not None else None, data_loaded=data_loaded,columns = columns,stock_name=stock_name,
                           count_data=count_data,num_columns=num_columns,file_path=file_path, filename=filename, global_name=global_name,output_columns=output_columns)

   
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
    testScore_rmse = 0
    testScore_mae = 0
    testScore_mse_real = 0
    testScore_rmse_real = 0
    testScore_mae_real = 0
    model_path_apple = 'Model/Apple/'
    model_path_amazon = 'Model/Amazon/'
    model_path_google = 'Model/Google/'
    algorithm = request.form.get('algorithm')
    column_prediction = request.form.get('column_prediction')
    useExistingModel = request.form.get('useExistingModel') 
    model = None
    scaled_data = None  
    scaler = None 
    start_train = 0
    end_train = 0
    time_train = 0
    start_predict = 0
    end_predict = 0    
    time_predict = 0 
    if  algorithm == 'algorithm-ffnn':       
        if useExistingModel == 'on': # For existing model
            default_hidden_layers = 1
            default_split_ratio = 0.8             
            if global_name == 'GOOGLE':
                train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))               
            elif global_name == 'APPLE':
                scaled_data, scaler = scale_data(global_data[column_prediction].values.reshape(-1,1))
                train, test = split_data_default(scaled_data)
                if column_prediction == 'Open': 
                    # Default parameters
                    default_hidden_neurons = 16
                    default_seq_size = 18
                    default_epochs = 400
                    default_batch_size = 32                
                    # Get parameters
                    split_ratio,hidden_neurons,seq_size,epochs,batch_sizes,hidden_layers = get_param_ffnn(default_split_ratio,default_hidden_neurons,default_seq_size,default_epochs,default_batch_size,default_hidden_layers)             
                    # Check if the user wants to use an existing model
                    model_path = model_path_apple + 'FFNN/FFNN_Model_Apple_Open.h5'
                    
                    start_train = time.time()
                    model = model_ffnn_exist(default_seq_size, default_hidden_neurons, model_path)
                    end_train = time.time()
                    
                    x,y = to_sequences(test,1,18)
                    start_predict = time.time()
                    test_pred = model.predict(x)
                    end_predict = time.time()
                    testScore_mse, testScore_rmse, testScore_mae = calculate_metrics(y, test_pred)
                    
                    y_real, test_pred_real = scale_data_original(y,test_pred,scaler)                                         
                    testScore_mse_real, testScore_rmse_real, testScore_mae_real = calculate_metrics(y_real, test_pred_real)
                    
                    time_train = round(end_train - start_train, 5)
                    time_predict = round(end_predict - start_predict, 5)
                    eda_model_FFNN(y,test_pred,column_prediction)
                    return jsonify(algorithm=algorithm, column_prediction=column_prediction, 
                                    testScore_mse=testScore_mse, testScore_rmse = testScore_rmse,testScore_mae = testScore_mae,
                                    testScore_mse_real=testScore_mse_real, testScore_rmse_real = testScore_rmse_real,testScore_mae_real = testScore_mae_real,
                                    hidden_neurons=default_hidden_neurons, seq_size=default_seq_size, 
                                    hidden_layers=default_hidden_layers, epochs=default_epochs, 
                                    batch_sizes=default_batch_size, split_ratio=default_split_ratio,
                                    time_train = time_train,time_predict = time_predict)                                      
            elif global_name == 'AMAZON':
                train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))
            elif global_name == 'Weather_WS':
                train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))    
            elif global_name == 'weather-HCM':
                train, test = split_data_default(scale_data(global_data[column_prediction].values.reshape(-1,1)))     
        else: # For new dataset or train no existing model
            split_ratio_new,hidden_neurons_new,seq_size_new,epochs_new,batch_sizes_new,hidden_layers_new = get_param_ffnn_datasetNew()            
            scaled_data, scaler = scale_data(global_data[column_prediction].values.reshape(-1,1))
            train_datanew, test_datanew = split_data_new(scaled_data,split_ratio_new)
            start_train = time.time()
            model = model_ffnn_new(train_datanew, test_datanew,hidden_layers_new, seq_size_new, hidden_neurons_new, epochs_new, batch_sizes_new)
            end_train = time.time()
            x,y = to_sequences(test_datanew,1,seq_size_new)      
            
            start_predict = time.time()
            test_pred = model.predict(x)
            end_predict = time.time()
            
            testScore_mse, testScore_rmse, testScore_mae = calculate_metrics(y, test_pred)
            
            y_real, test_pred_real = scale_data_original(y,test_pred,scaler)                                         
            testScore_mse_real, testScore_rmse_real, testScore_mae_real = calculate_metrics(y_real, test_pred_real)
            
            eda_model_FFNN(y,test_pred,column_prediction)
            
            time_train = round(end_train - start_train, 5)
            time_predict = round(end_predict - start_predict, 5)
            return jsonify(algorithm=algorithm, column_prediction=column_prediction, 
                           testScore_mse=testScore_mse,testScore_rmse=testScore_rmse,testScore_mae=testScore_mae,
                           testScore_mse_real=testScore_mse_real, testScore_rmse_real = testScore_rmse_real,testScore_mae_real = testScore_mae_real,
                           hidden_neurons=hidden_neurons_new, seq_size=seq_size_new, hidden_layers=hidden_layers_new, 
                           epochs=epochs_new, batch_sizes=batch_sizes_new, split_ratio=split_ratio_new,
                           time_train = time_train,time_predict = time_predict)     
    elif algorithm == 'algorithm-lstm':
        if useExistingModel == 'on': 
            array_stock = list(["Open","High","Low","Close","Adj Close"])
            epochs_lstm = 300
            batch_sizes_lstm = 16
            hidden_layers_lstm = 1
            split_ratio_lstm = 0.8
            
            if global_name == 'APPLE':
                seq_size_lstm = 12
                hidden_neurons_lstm = 5
                output_lstm = 5      
                model_path_lstm = model_path_apple + 'LSTM/LSTM_APPLE.h5'
                scaled_data, scaler = scale_data(global_data[array_stock])
                train, test = split_data_default(scaled_data)    
                
                start_train = time.time()         
                model_lstm = LSTM_exist(train, output_lstm, seq_size_lstm, hidden_neurons_lstm, model_path_lstm)
                end_train = time.time()
                
                testX_LSTM, testY_LSTM = to_sequences_multivariate_lstm(test,seq_size_lstm)           
                
                start_predict = time.time()
                result_LSTM = model_lstm.predict(testX_LSTM) 
                end_predict = time.time()
                
                predict_LSTM_real = scaler.inverse_transform(result_LSTM)
                textY_LSTM_real = scaler.inverse_transform(testY_LSTM)
                time_train = round(end_train - start_train, 5)
                time_predict = round(end_predict - start_predict, 5) 
                                                             
                testScore_mse, testScore_rmse, testScore_mae, testScore_mse_real, testScore_rmse_real, testScore_mae_real = LSTM_Predict(result_LSTM,testY_LSTM,predict_LSTM_real,textY_LSTM_real,array_stock,column_prediction)
                
                return jsonify(algorithm=algorithm, column_prediction=column_prediction,
                                testScore_mse=testScore_mse,testScore_rmse=testScore_rmse,testScore_mae=testScore_mae,
                                testScore_mse_real=testScore_mse_real, testScore_rmse_real = testScore_rmse_real,testScore_mae_real = testScore_mae_real,
                                hidden_neurons = hidden_neurons_lstm, 
                                seq_size=seq_size_lstm, 
                                hidden_layers= hidden_layers_lstm,
                                epochs=epochs_lstm, 
                                batch_sizes=batch_sizes_lstm, 
                                split_ratio=split_ratio_lstm,
                                time_train = time_train,time_predict = time_predict)
            elif global_name == 'GOOGLE':
                seq_size_lstm = 12
                hidden_neurons_lstm = 5
                output_lstm = 5      
                model_path_lstm = model_path_google + 'LSTM/LSTM_GOOGLE.h5'
                scaled_data, scaler = scale_data(global_data[array_stock])
                train, test = split_data_default(scaled_data)    
                
                start_train = time.time()         
                model_lstm = LSTM_exist(train, output_lstm, seq_size_lstm, hidden_neurons_lstm, model_path_lstm)
                end_train = time.time()
                
                testX_LSTM, testY_LSTM = to_sequences_multivariate_lstm(test,seq_size_lstm)           
                
                start_predict = time.time()
                result_LSTM = model_lstm.predict(testX_LSTM) 
                end_predict = time.time()
                
                predict_LSTM_real = scaler.inverse_transform(result_LSTM)
                textY_LSTM_real = scaler.inverse_transform(testY_LSTM)
                time_train = round(end_train - start_train, 5)
                time_predict = round(end_predict - start_predict, 5) 
                                                             
                testScore_mse, testScore_rmse, testScore_mae, testScore_mse_real, testScore_rmse_real, testScore_mae_real = LSTM_Predict(result_LSTM,testY_LSTM,predict_LSTM_real,textY_LSTM_real,array_stock,column_prediction)
                
                return jsonify(algorithm=algorithm, column_prediction=column_prediction,
                                testScore_mse=testScore_mse,testScore_rmse=testScore_rmse,testScore_mae=testScore_mae,
                                testScore_mse_real=testScore_mse_real, testScore_rmse_real = testScore_rmse_real,testScore_mae_real = testScore_mae_real,
                                hidden_neurons = hidden_neurons_lstm, 
                                seq_size=seq_size_lstm, 
                                hidden_layers= hidden_layers_lstm,
                                epochs=epochs_lstm, 
                                batch_sizes=batch_sizes_lstm, 
                                split_ratio=split_ratio_lstm,
                                time_train = time_train,time_predict = time_predict)                
            elif global_name == 'AMAZON':
                seq_size_lstm = 12
                hidden_neurons_lstm = 5
                output_lstm = 5      
                model_path_lstm = model_path_amazon + 'LSTM/LSTM_AMAZON.h5'
                scaled_data, scaler = scale_data(global_data[array_stock])
                train, test = split_data_default(scaled_data)    
                
                start_train = time.time()         
                model_lstm = LSTM_exist(train, output_lstm, seq_size_lstm, hidden_neurons_lstm, model_path_lstm)
                end_train = time.time()
                
                testX_LSTM, testY_LSTM = to_sequences_multivariate_lstm(test,seq_size_lstm)           
                
                start_predict = time.time()
                result_LSTM = model_lstm.predict(testX_LSTM) 
                end_predict = time.time()
                
                predict_LSTM_real = scaler.inverse_transform(result_LSTM)
                textY_LSTM_real = scaler.inverse_transform(testY_LSTM)
                time_train = round(end_train - start_train, 5)
                time_predict = round(end_predict - start_predict, 5) 
                                                             
                testScore_mse, testScore_rmse, testScore_mae, testScore_mse_real, testScore_rmse_real, testScore_mae_real = LSTM_Predict(result_LSTM,testY_LSTM,predict_LSTM_real,textY_LSTM_real,array_stock,column_prediction)
                
                return jsonify(algorithm=algorithm, column_prediction=column_prediction,
                                testScore_mse=testScore_mse,testScore_rmse=testScore_rmse,testScore_mae=testScore_mae,
                                testScore_mse_real=testScore_mse_real, testScore_rmse_real = testScore_rmse_real,testScore_mae_real = testScore_mae_real,
                                hidden_neurons = hidden_neurons_lstm, 
                                seq_size=seq_size_lstm, 
                                hidden_layers= hidden_layers_lstm,
                                epochs=epochs_lstm, 
                                batch_sizes=batch_sizes_lstm, 
                                split_ratio=split_ratio_lstm,
                                time_train = time_train,time_predict = time_predict)            
        else: # Train a new model or use a new dataset
            global array_column_new
            output_lstm_new = len(array_column_new)
            split_ratio_lstm_new, hidden_neurons_lstm_new,seq_size_lstm_new, epochs_lstm_new, batch_sizes_lstm_new, hidden_layers_lstm_new = get_param_ffnn_datasetNew()                                          
            
            scaled_data, scaler = scale_data(global_data[array_column_new])
            train_new, test_new  = split_data_new(scaled_data,split_ratio_lstm_new)
            
            start_train = time.time()
            model_new = LSTM_new(train_new,test_new,output_lstm_new, seq_size_lstm_new, hidden_neurons_lstm_new,epochs_lstm_new,batch_sizes_lstm_new, hidden_layers_lstm_new)
            end_train = time.time()
            
            testX_LSTM_new, testY_LSTM_new = to_sequences_multivariate_lstm(test_new, seq_size_lstm_new)                               
            
            start_predict = time.time()
            result_LSTM_new = model_new.predict(testX_LSTM_new) 
            end_predict = time.time()
            
            predict_LSTM_real = scaler.inverse_transform(result_LSTM_new)
            textY_LSTM_real = scaler.inverse_transform(testY_LSTM_new)   
            testScore_mse, testScore_rmse, testScore_mae, testScore_mse_real, testScore_rmse_real, testScore_mae_real = LSTM_Predict(result_LSTM_new,testY_LSTM_new,predict_LSTM_real,textY_LSTM_real,array_column_new,column_prediction)
            time_train = round(end_train - start_train, 5)
            time_predict = round(end_predict - start_predict, 5)
                        
            return jsonify(algorithm=algorithm, column_prediction=column_prediction, 
                           testScore_mse=testScore_mse, testScore_rmse=testScore_rmse,testScore_mae=testScore_mae,  
                           testScore_mse_real=testScore_mse_real, testScore_rmse_real = testScore_rmse_real,testScore_mae_real = testScore_mae_real,
                           hidden_layers=hidden_layers_lstm_new,
                           hidden_neurons=hidden_neurons_lstm_new, 
                           seq_size=seq_size_lstm_new,
                           epochs=epochs_lstm_new, 
                           batch_sizes=batch_sizes_lstm_new, 
                           split_ratio=split_ratio_lstm_new,
                           time_train = time_train,time_predict = time_predict)
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
    return jsonify({'message': 'Received'}), 200

array_column_new = []
@app.route('/getcolumn_ouput_multi', methods=['POST'])
def get_columns():
    global array_column_new
    array_column_new = request.get_json()
    print(array_column_new)
    return jsonify({'message': 'Received'}), 200
    
if __name__ == '__main__':
    app.run(debug=True) 