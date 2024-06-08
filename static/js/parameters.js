var varnnParameters = ['Parameter 1', 'Parameter 2', 'Parameter 3'];
var ffnnParameters = ['Epoch', 'Batch size', 'Data window size', 'Hidden Neurons', 'Hidden Layers'];
var lstmParameters = ['Epoch', 'Batch size', 'Data window size', 'Hidden Neurons', 'Hidden Layers'];
var varParameters = ['Parameter A12', 'Parameter B23', 'Parameter A11', 'Parameter B22'];
var arimaParameters = ['order of autoregression(p)', 'degree of differencing(d)', 'order of moving average(q)'];


function createAlgorithmParameters() {
    return {
        'algorithm-varnn': varnnParameters,
        'algorithm-ffnn': ffnnParameters,
        'algorithm-lstm': lstmParameters,
        'algorithm-var': varParameters,
        'algorithm-arima': arimaParameters,
    };
}

var algorithmParameters = {
    'APPLE': {
        'Open': createAlgorithmParameters(),
        'High': createAlgorithmParameters(),
        'Low': createAlgorithmParameters(),
        'Close': createAlgorithmParameters(),
        'Adj Close': createAlgorithmParameters()
    },
    'GOOGLE': {
        'Open': createAlgorithmParameters(),
        'High': createAlgorithmParameters(),
        'Low': createAlgorithmParameters(),
        'Close': createAlgorithmParameters(),
        'Adj Close': createAlgorithmParameters()
    },
    'AMAZON': {
        'Open': createAlgorithmParameters(),
        'High': createAlgorithmParameters(),
        'Low': createAlgorithmParameters(),
        'Close': createAlgorithmParameters(),
        'Adj Close': createAlgorithmParameters()
    },
    'Weather_WS': {
        'Pressure': createAlgorithmParameters(),
        'Temperature': createAlgorithmParameters(),
        'Saturation_vapor_pressure': createAlgorithmParameters(),
        'Vapor_pressure_deficit': createAlgorithmParameters(),
        'Specific_humidity': createAlgorithmParameters(),
        'Airtight': createAlgorithmParameters(),
        'Wind_speed': createAlgorithmParameters()
    },
    'weather-HCM': {
        'max': createAlgorithmParameters(),
        'min': createAlgorithmParameters(),
        'wind': createAlgorithmParameters(),
        'rain': createAlgorithmParameters(),
        'humidi': createAlgorithmParameters(),
        'pressure': createAlgorithmParameters()
    }
}

