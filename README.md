# RNN-Based Stock Price Prediction

This project uses a Recurrent Neural Network (RNN) to predict the **'Close' prices** of four major tech companies — `AMZN`, `MSFT`, `GOOGL`, and `IBM` — using historical stock data.

---

## Dataset

Stock data for:
- Amazon (AMZN)
- Microsoft (MSFT)
- Google (GOOGL)
- IBM

Each CSV file contains historical daily stock information including `Open`, `High`, `Low`, `Close`, `Volume`, and `Adj Close`.

---

## Tools & Libraries

- **Python 3.8+**
- **Pandas**, **NumPy** – data manipulation
- **Matplotlib**, **Seaborn** – data visualization
- **scikit-learn** – scaling and preprocessing
- **TensorFlow / Keras** – building and training RNN model

---

## Project Steps

### 1. Data Preparation
- Loaded and cleaned 4 stock CSVs
- Combined into a master DataFrame
- Extracted `Close` prices for target prediction
- Windowed dataset:  
  - `window_size = 65`  
  - `stride = 5`  
  - `test_size = 20%`

### 2. Exploratory Data Analysis
- Frequency distributions of volumes
- Volume variation over time
- Correlation between stock features

### 3. Data Preprocessing
- MinMax Scaling applied to all features
- Time-series windowing for `X` and multi-output `y`

### 4. Model Architecture
Custom RNN model built using `Keras`:
- Multiple RNN layers
- Dropout regularization
- Output layer with 4 neurons for 4 stock targets

### 5. Hyperparameter Tuning
Performed **Manual Grid Search**:
- `128` RNN units  
- `2` layers  
- `0.3` dropout  
- `rmsprop` optimizer with `0.001` learning rate  
- `batch size` of 32, `10` epochs`

### 6. Final Model Results
```
Final Test Loss (MSE): 46493.808594
Final Test MAE: 210.395966

```

> *(Numbers will vary slightly based on final training run)*

---

### 7. Visualizations

- Volume trends
- Correlation heatmap
- Predicted vs Actual 'Close' prices
- Training vs Validation Loss

---

### 8. Acknowledgements
Developed as part of the assignment for Topic about Recurrent Neural Networks for Executive Post Graduate Program in Machine Learning and AI - IIIT,Bangalore.

### 9. Contact
- Pankaj Kumar Agrawal  Email: pankaj8blr@gmail.com