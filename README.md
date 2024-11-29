# Multi Dataset Time Series Models
A comprehensive project for time series forecasting using the VARNN (Vector AutoRegression Neural Network) model. This project includes a comparison with VAR, FFNN, ARIMA, and LSTM models across univariate and multivariate time series tasks, alongside an interactive web demo.

In this study, the experimental data consists of six datasets: https://github.com/lonGDiBo/Multi_Dataset_Time_Series_Models/tree/main/dataset
- Two datasets related to weather.
- Three datasets containing stock prices from three different corporations. The stock datasets include data for APPLE, AMAZON, and GOOGLE, collected from the Yahoo Finance website using the Python library yfinance.
- The final dataset provides an overview of Vietnam's macroeconomic indicators.

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
### Theoretical 
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

### Model Training Process
To determine the best VARNN model, the training process follows these steps:

- Split the Dataset: Divide the dataset into a training set (to train the model) and a test set (to evaluate model performance).
Identify the Appropriate Lag

- Determine the optimal lag, which represents the number of previous time steps of the dependent variable used as model input.Select the Number of Neurons in the Hidden Layer

- Choose the appropriate number of neurons based on factors such as the complexity of the problem and the amount of available data. Randomly Initialize Weights

- Randomly initialize weights for connections between the input and hidden layers, as well as between the hidden and output layers. This step is crucial to start the training process. Compute Outputs for Hidden Neurons

- Use the sigmoid logistic activation function to calculate the hidden layer outputs. Pass the inputs through connections to the hidden layer and apply the activation function for nonlinear transformations. Compute Outputs for Output Neurons

- Use the VAR model formula to compute the outputs of the output neurons. Combine the results from the hidden layer and apply the VAR formula for linear computations. Calculate Error Gradients for Output Neurons

- Compute the error gradient for each output neuron based on the difference between the predicted output and the actual target value. Adjust Weights in the Output Layer

- Update the output layer weights using the error gradient and the learning rate. This step adjusts the weights to minimize prediction errors. Update All Weights in the Output Layer

- Apply the computed weight adjustments to update all weights in the output layer. Calculate Error Gradients for Hidden Neurons

- Compute the error gradient for each hidden neuron based on the error contributions from the output layer and the connection weights. Adjust Weights in the Hidden Layer

- Update the hidden layer weights using the error gradient and the learning rate. This step reduces errors propagated from the output layer to the hidden layer. Update All Weights in the Hidden Layer

- Apply the computed weight adjustments to update all weights in the hidden layer. Evaluate Model Accuracy

- Use evaluation metrics such as Mean Absolute Percentage Error (MAPE) and Mean Squared Error (MSE) to assess the accuracy of the VARNN model. This step identifies the best-performing model based on the chosen evaluation criteria. Make Predictions with the Trained Model

- Once the best model is identified, use it to make predictions on new, unseen data. Pass the input data through the trained model, following the described steps, to obtain predictions.

## 4. How to run code
### Run Locally 
 - Clone the repository:
   - ```git clone https://github.com/lonGDiBo/Multi_Dataset_Time_Series_Models.git```
   - ```cd your-repo-name```
- Install dependencies:
  - ```pip install -r requirements.txt```
- Start the application:
  - ```python app.py```
- Open your browser and navigate to: http://localhost:5000. (flask run port 5000) 

### 5. How to Use the Web Prediction Application
The time series prediction application is built using the Flask framework. It allows users to make predictions on datasets with optimized model parameters, and models are saved in h5 format. Additionally, users can upload their own datasets from personal devices to perform time series predictions using algorithms such as VARNN, FFNN, LSTM, VAR, and ARIMA.

- Click on the "Upload" button to upload your dataset or  select from preloaded datasets that have been trained with the best-optimized parameters for each model. This allows for quick and efficient predictions without additional configuration.
- Ensure the dataset follows the required format (e.g., CSV with proper headers).
![image](https://github.com/user-attachments/assets/6c61f50a-9a8c-4f0f-972b-f06b6c705a54)

-When the user clicks the "Load Data" button, the application performs the following data preprocessing steps to ensure accurate and efficient predictions by the algorithms:
  - Removes invalid values from columns with data types string and datetime.
  - Handles NaN (Not a Number) values appropriately.
This ensures the dataset is clean and ready for prediction.
![image](https://github.com/user-attachments/assets/329c67bc-9db1-4e15-b3e0-e62bc1666d52)

- After the data is successfully loaded, users can view information about the dataset and explore trends of the attributes through line charts.

- Users can select the specific columns they want to use for prediction and choose the desired algorithm (e.g., VARNN, FFNN, LSTM, VAR, or ARIMA) for the forecasting process.
  
![image](https://github.com/user-attachments/assets/aaa0e386-4d65-4c05-8e00-e1bcc55b7dde)
![image](https://github.com/user-attachments/assets/3d7e1524-a9eb-47d3-b065-bb9819930912)


- Users can perform predictions on the entire test dataset or specify a smaller sequence for forecasting. The model uses optimized parameters by default, but users can also manually input custom parameter values if needed.
  
![image](https://github.com/user-attachments/assets/6c09d39d-8f8d-4a0d-910d-7b5250e4215d)

- After successfully performing predictions, users can view evaluation metrics such as MSE, RMSE, and MAE for both normalized and actual data values. Additionally, users can visualize comparison charts showing the actual vs. predicted values based on the selected algorith
  
![image](https://github.com/user-attachments/assets/a1cf773f-23bf-4b38-b73a-4e4d1b7cf10d)







