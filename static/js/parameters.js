
var varnnParameters = ['Epoch', 'Batch size', 'Lag order(p)', 'Hidden Neurons'];
var ffnnParameters = ['Epoch', 'Batch size', 'Data window size', 'Hidden Neurons', 'Hidden Layers'];
var lstmParameters = ['Epoch', 'Batch size', 'Data window size', 'Hidden Neurons', 'Hidden Layers'];
var varParameters = ['Max lag order(p)'];
var arimaParameters = ['Autoregressive order (p)', 'Differencing order (d)', 'Moving average order (q)'];

var defaultParameters = {
    'algorithm-varnn': varnnParameters,
    'algorithm-ffnn': ffnnParameters,
    'algorithm-lstm': lstmParameters,
    'algorithm-var': varParameters,
    'algorithm-arima': arimaParameters,
};

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

