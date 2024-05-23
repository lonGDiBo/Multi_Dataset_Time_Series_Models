from flask import Flask
from flask import *
from markupsafe import escape
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)



from flask import abort, redirect, url_for


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def eda_stocks(dataset,column_name):
    data = dataset.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    plt.clf() 
    data[column_name].plot(color = 'blue')
    plt.title('Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.savefig('static/images/plot.png')  # Save the plot as an image
    plt.close()
    
    
    
@app.route('/')
def index():
    return render_template('login.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    data_loaded = False
    stock_name = None
    column_name = None
    data = None
    columns = None

    if request.method == 'POST':
        if 'data' in request.form:
            stock_name = request.form['data']
            file_path = f"{stock_name}.csv"
            data = load_data(f"./data/{file_path}")
            data = data.head(10)
            data_loaded = True
            columns = data.columns.tolist()

        if 'column' in request.form:
            column_name = request.form.get('column')

    return render_template('main.html', data=data.to_html() if data is not None else None, columns=columns, data_loaded=data_loaded, stock_name=stock_name, column_name=column_name)


@app.route('/get_column_data', methods=['POST'])
def get_column_data():
    stock_name = request.form['stock_name']
    column_name = request.form['column_name']
    file_path = f"./data/{stock_name}.csv"
    data = load_data(file_path)
    eda_stocks(data, column_name)
    return redirect(url_for('data'))
    