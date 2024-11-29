# Multi Dataset Time Series Models
A comprehensive project for time series forecasting using the VARNN (Vector AutoRegression Neural Network) model. This project includes a comparison with VAR, FFNN, ARIMA, and LSTM models across univariate and multivariate time series tasks, alongside an interactive web demo.
## 1. Introduction
### Objective:
- Develop and implement the VARNN (Vector AutoRegression Neural Network) model, a hybrid approach combining statistical methods (VAR) with the power of neural networks (FFNN).
- Perform a comparative analysis of VARNN against VAR, FFNN, ARIMA, and LSTM models for both univariate and multivariate time series forecasting.
- Provide an interactive web demo for users to experiment with these models and evaluate their performance on custom datasets.

### Core Features:
- Integration of VAR for capturing multivariate relationships and FFNN for learning complex nonlinear patterns.
- Emphasis on the strengths and weaknesses of each model type: statistical, neural network, and hybrid.
- Support for multiple datasets, making the project versatile for various domains like energy forecasting, financial markets, and weather prediction.
- Metrics such as MAE, RMSE, and MAPE are used for evaluation.
### Technologies Used:
- Python, Flask, scikit-learn, TensorFlow, and others
  
## 2. Model

- VARNN: Combines VAR and FFNN for hybrid forecasting.
- VAR: Traditional statistical model for multivariate time series.
- FFNN: Feedforward Neural Network for nonlinear univariate and multivariate forecasting.
- ARIMA: A classic univariate statistical model for time series forecasting.
- LSTM: Recurrent Neural Network (RNN) specialized for sequence modeling.

## 3. VARNN Model
![VARNN model architecture](https://github.com/user-attachments/assets/e8bd9fc5-b924-40a4-b5c5-30454717d9df)

In a multivariate time series with 𝑚 m variables influenced by 𝑝 p lags, the input vector is defined as:
![image](https://github.com/user-attachments/assets/e7fe70a4-927a-48d2-bcd0-df4bb8ad694b)

The number of neurons in the input layer is 𝑝 × 𝑚 p×m, where 𝑝 p is the number of time lags and 𝑚 m is the number of variables. For a hidden layer with ℎ h hidden units, the weight matrix connecting the input layer to the hidden layer will have dimensions ( 𝑝 × 𝑚 ) × ℎ (p×m)×h.

The neurons in the input layer remain fixed in the architecture and are connected to every neuron in both the hidden layer and the output layer. This results in biases: 

𝛼 = ( 𝛼 1 , 𝛼 2 , … , 𝛼 ℎ ) ′ α=(α 1 ​ ,α 2 ​ ,…,α h ​ ) ′ for the hidden layer. 

𝛽= ( 𝛽 1 , 𝛽 2 , … , 𝛽 𝑚 ) β=(β 1 ​ ,β 2 ​ ,…,β m ​ ) for the output layer.

The final formula for calculating the output of the VARNN model is defined as:

![image](https://github.com/user-attachments/assets/e7f6e729-05c6-4342-b8e2-a4aa6d926a2f)

The detailed formula for the output of the VARNN model can be expressed as:

![image](https://github.com/user-attachments/assets/27f671b5-14c2-4f1d-8ed3-d42a52300999)
 
 where
 - zt: Output value for variable
 - 𝜆: Weights of the output layer.
 - F(*): Activation function of the hidden layer.
 - 𝛼: Bias of the hidden layer.
 - 𝛽: Bias of the output layer.
 - ɛ: Error term (noise).

## 4. How to use
### Run Locally 
 - Clone the repository:
   - ```git clone https://github.com/lonGDiBo/Multi_Dataset_Time_Series_Models.git```
   - ```cd your-repo-name```
- Install dependencies:
  - ```pip install -r requirements.txt```
- Start the application:
  - ```python app.py```
- Open your browser and navigate to: http://localhost:5000. (flask run port 5000) 




