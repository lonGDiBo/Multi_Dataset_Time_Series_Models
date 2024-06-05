var algorithmParameters = {
    'APPLE': {
        'Open':
        {
            'algorithm-varnn': ['Parameter 1', 'Parameter 2', 'Parameter 3'],
            'algorithm-ffnn': ['Epoch', 'Batch size', 'Data window size', 'Hidden Neurons', 'Hidden Layers'],
            'algorithm-lstm': ['Parameter A1'],
            'algorithm-var': ['Parameter A12', 'Parameter B23', 'Parameter A11', 'Parameter B22'],
            'algorithm-arima': ['order of autoregression(p)', 'degree of differencing(d)', 'order of moving average(q)'],

        }
    },
    'GOOGLE': {
        'Open':
        {
            'algorithm-varnn': ['Parameter 1', 'Parameter 2', 'Parameter 3'],
            'algorithm-ffnn': ['Epoch', 'Batch size', 'Data window size', 'Hidden Neurons', 'Hidden Layers'],
            'algorithm-lstm': ['Parameter A1'],
            'algorithm-var': ['Parameter A12', 'Parameter B23', 'Parameter A11', 'Parameter B22'],
            'algorithm-arima': ['order of autoregression(p)', 'degree of differencing(d)', 'order of moving average(q)'],

        }
    }
};

var defaultValues = {
    'APPLE': {
        'Open': {
            'algorithm-varnn': {
                'Parameter 1': 'value 1',
                'Parameter 2': 'value 2',
                'Parameter 3': 'value 3'
            },
            'algorithm-ffnn': {
                'Epoch': '100',
                'Batch size': '32',
                'Data window size': '18',
                'Hidden Neurons': '16',
                'Hidden Layers': '2'
            },
            'algorithm-lstm': {
                'Parameter A1': 'value A1'
            },
            'algorithm-var': {
                'Parameter A12': 'value A12',
                'Parameter B23': 'value B23',
                'Parameter A11': 'value A11',
                'Parameter B22': 'value B22'
            },
            'algorithm-arima': {
                'order of autoregression(p)': 'value p',
                'degree of differencing(d)': 'value d',
                'order of moving average(q)': 'value q'
            }
        }
    },
    'GOOGLE': {
        'Open': {
            'algorithm-varnn': {
                'Parameter 1': 'value 1',
                'Parameter 2': 'value 2',
                'Parameter 3': 'value 3'
            },
            'algorithm-ffnn': {
                'Epoch': '100',
                'Batch size': '34',
                'Data window size': '18',
                'Hidden Neurons': '20',
                'Hidden Layers': '2'
            },
            'algorithm-lstm': {
                'Parameter A1': 'value A1'
            },
            'algorithm-var': {
                'Parameter A12': 'value A12',
                'Parameter B23': 'value B23',
                'Parameter A11': 'value A11',
                'Parameter B22': 'value B22'
            },
            'algorithm-arima': {
                'order of autoregression(p)': 'value p',
                'degree of differencing(d)': 'value d',
                'order of moving average(q)': 'value q'
            }
        }
    }
};