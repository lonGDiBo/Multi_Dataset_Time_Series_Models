# VARNN for Time Series Prediction
A comprehensive project for time series forecasting using the VARNN (Vector AutoRegression Neural Network) model. This project includes a comparison with VAR, FFNN, ARIMA, and LSTM models across univariate and multivariate time series tasks, alongside an interactive web demo.

In this study, the experimental data consists of six datasets: https://github.com/lonGDiBo/Multi_Dataset_Time_Series_Models/tree/main/dataset
- Two datasets related to weather.()
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
- Metrics such as MAE, RMSE, and MSE are used for evaluation.
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

### How to Select Variables for Prediction in Multivariate Models
- Domain Knowledge: Use expert knowledge to identify variables likely to affect the target variable.
- Correlation Analysis: Identify variables strongly correlated with the target and remove redundant predictors.
- Granger Causality Test: Select variables with significant temporal relationships (p-value < 0.05).
- Feature Importance: Use ML models (e.g., Random Forest) to rank variables by importance.
- Lagged Variables: Include optimal time lags of predictors and the target using AIC/BIC or PACF.
- Dimensionality Reduction: Use PCA or autoencoders to reduce variables while preserving key information.
- Regularization: Apply Lasso or Elastic Net to select relevant variables and handle multicollinearity.
- Cross-Validation: Test variable combinations and retain those consistently improving accuracy.

## 4. Result

All models in the project are made with the best optimized parameters to ensure fair comparisons and avoid the influence of skewed factors.

Model Performance on Datasets
After conducting experiments across six datasets (two weather, three stock prices, and one macroeconomic dataset), the following results were obtained:
### Evaluation Metrics:
- Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and  Mean Squared Error (MSE) were used to evaluate the models.
- Normalized and actual data values were considered for robust comparisons.
- Execution time

### Model Comparisons:
- VARNN outperformed traditional models like VAR and ARIMA in capturing nonlinear relationships in multivariate datasets.
- LSTM performed comparably to VARNN for longer sequences but required more computational resources and tuning.
- FFNN was effective for simple univariate tasks but struggled with multivariate scenarios.
- ARIMA showed strong results for stationary, univariate data but was less effective for datasets with high variability.

### Detailed Model Performance
The table below presents the detailed performance metrics (MSE, RMSE, MAE) for the models on both normalized and actual values, as well as their training and testing times:
#### * STOCK data
##### ** AMAZON Dataset
Metric | VARNN | FFNN | ARIMA | VAR | LSTM
--- | --- | --- | --- |--- |---
MSE | 0.0002528 | 0.0003189 | 0.0003343 | 0.0002821 | 0.0002554
RMSE | 0.0159 | 0.01785 | 0.01852 | 0.01679 | 0.01598
MAE | 0.01133 | 0.01326 | 0.01333 | 0.01164 | 0.01158
MSE (Actual Values) | 6.685 | 8.440 | 9.079 | 7.458 | 6.754
RMSE (Actual Values) | 2.585 | 2.905 | 3.013 | 2.730 | 2.598
MAE (Actual Values) | 1.843 | 2.158 | 2.168 | 1.865 | 1.884
Training Time (s) | 57.333 | 976.57 | 71.5 | 10.6 | 2142.258
Testing Time (s) | 0.248 | 0.642 | 224.012 | 0.01 | 0.522

- Compare each variable of the VARNN model with other models
  
![image](https://github.com/user-attachments/assets/9428c4df-7803-4b37-a803-b42de829dadf)

##### ** APPLE Dataset
Metric | VARNN | FFNN | ARIMA | VAR | LSTM
--- | --- | --- | --- |--- |---
MSE	| 0.0001938	| 0.0002472  |	0.000244 |	0.000187 |	0.0001776
RMSE	| 0.01392 |	0.01572 |	0.01562 |	0.0137 |	0.01332
MAE |	0.01048 |	0.01199 |	0.01176 |	0.00997 |	0.00985
MSE (Actual Values)	| 5.892 |	7.501 |	7.385 |	5.704 |	5.397
RMSE (Actual Values)|	2.427 |	2.738|	2.717|	2.388|	2.323
MAE (Actual Values)|	1.827|	2.089|	2.045|	1.737|	1.716
Training Time (s)|	66.11	|858.271|	58.613|	8.003|	2139.778
Testing Time (s)|	0.188	|0.655|	176.036|	0.0108|	0.457

- Compare each variable of the VARNN model with other model
  
![image](https://github.com/user-attachments/assets/772509ab-6782-4379-994e-0f539d3271b4)


##### ** GOOGLE Dataset
Metric | VARNN | FFNN | ARIMA | VAR | LSTM
--- | --- | --- | --- |--- |---
MSE	| 0.0003305	| 0.0003865|	0.000384	|0.000316	|0.0003273
RMSE	| 0.01818|	0.01966	|0.01959	|0.01777|	0.01809
MAE	|0.01334	|0.01469	|0.01461|	0.01257	|0.01292
MSE (Actual Values)|	4.565	|5.434	|5.312	|4.361|	4.523
RMSE (Actual Values)|	2.136|	2.331|	2.304	|2.088|	2.126
MAE (Actual Values)|	1.569|	1.728	|1.719	|1.478|	1.519
Training Time (s)|	62.996	|714.748|	62.182|	0.0075|	283.026
Testing Time (s)|	0.134	|0.996	|165.392|	0.021	|0.724

- Compare each variable of the VARNN model with other models
  
![image](https://github.com/user-attachments/assets/4272a193-d036-4960-8255-01b43928753d)

#### * Weather data
##### ** HoChiMinh wheather dataset
Metric | VARNN | FFNN | ARIMA | VAR | LSTM
--- | --- | --- | --- |--- |---
MSE	|0.004688	|0.004859	|0.00641|	0.004707|	0.005544
RMSE	|0.06847|	0.06971|	0.08006|	0.06861	|0.07446
MAE	|0.04821	|0.04853|	0.05763	|0.04813|	0.05275
MSE (Actual Values)|	17.431	|18.234 | 22.137	17.464	|18.548
RMSE (Actual Values)|	4.175|	4.270	|4.704	|4.179	|4.306
MAE (Actual Values)|	2.047	||2.052	|3.805|	2.088|	2.227
Training Time (s)|	87.637	|799.312	|2921.789	|0.094|	4238.094
Testing Time (s)	|0.854|	7200	|0.058	|0.741|	0.264

- Compare each variable of the VARNN model with other models
  
![image](https://github.com/user-attachments/assets/319d4484-c787-47e5-8b28-b94da73cd86e)

##### ** WS Beutenberg wheather dataset
Metric | VARNN | FFNN | ARIMA | VAR | LSTM
--- | --- | --- | --- |--- |---
MSE	 |0.003258	 |0.003466 |	0.006359	 |0.003302 |	0.003208
RMSE	 |0.05707	 |0.05887 |	0.08265 |	0.05746	 |0.05664
MAE	 |0.03932	 |0.03977	 |0.06126	 |0.03942 |	0.03883
MSE (Actual Values) |	27.77	 |29.548 |	41.632	 |28.149 |	27.348
RMSE (Actual Values) |	5.108	 |5.435 |	6.452	 |5.177 |	5.030
MAE (Actual Values) |	2.692 |	2.865 |	3.399 |	2.729 |	2.525
Training Time (s) |	144.885	 |1744.04	 |3531.169 |	0.03	 |1887.808
Testing Time (s)	 |0.645	 |1.753 |	13320 |	0.067 |	0.698

- Compare each variable of the VARNN model with other models
  
![image](https://github.com/user-attachments/assets/32edb2e6-7acc-4449-b092-b875a8099509)

#### * Vietnam's macroeconomic dataset
Metric | VARNN | FFNN | ARIMA | VAR | LSTM
--- | --- | --- | --- |--- |---
MSE	 |0.00265 |	0.00323 |	0.00458 |	0.00328	 |0.00397
RMSE	 |0.05149	 |0.05687	 |0.06770	 |0.05727 |	0.06302
MAE	 |0.03773	 |0.04068	 |0.04394	 |0.04000 |	0.05035
MSE (Actual Values) |	1.705 |	1.95653 |	2.8318 |	2.03821	 |2.77341
RMSE (Actual Values)	 |1.305	 |1.398 |	1.682	 |1.665 |	1.427
MAE (Actual Values)	 |0.909	 |0.987	 |0.996 |	1.288 |	0.979
Training Time (s) |	21.12 |	61.22	 |4.323	 |0.154 |	46.667
Testing Time (s)	 |0.169	 |0.267	 |4.821 |	0.015 |	0.578

- Compare each variable of the VARNN model with other models
  
![image](https://github.com/user-attachments/assets/b81cceff-8621-4e53-b2c0-c5f529956e37)

### General Conclusion
- When comparing the VARNN model with univariate models, VARNN demonstrates superior performance in both efficiency and training/testing time. Univariate models require training each variable separately, which prevents interactions between variables and results in significantly weaker performance compared to VARNN.

- In comparison with multivariate models, VARNN occasionally underperforms on certain datasets. Specifically, the LSTM model outperforms VARNN in terms of accuracy on some datasets. However, the training and prediction time of VARNN is significantly shorter, making it more advantageous in scenarios where fast execution is required, and the accuracy only needs to meet acceptable thresholds.

- Additionally, VARNN shows a considerable advantage when working with datasets that have fewer observations. Its ability to achieve robust results on smaller datasets further solidifies its practicality and versatility in a wide range of time series forecasting tasks.

## 5. How to run code
### Run Locally 
 - Clone the repository:
   - ```git clone https://github.com/lonGDiBo/Multi_Dataset_Time_Series_Models.git```
   - ```cd your-repo-name```
- Install dependencies:
  - ```pip install -r requirements.txt```
- Start the application:
  - ```python app.py```
- Open your browser and navigate to: http://localhost:5000. (flask run port 5000) 

### 6. How to Use the Web Prediction Application
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


### Contact

For any inquiries or support, feel free to reach out at
- Email:  [longpm211@gmail.com].
- Linked: [longl](https://www.linkedin.com/in/minhlongde/)




