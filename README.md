# Team Details

24M2005 Prabhat Patel
24M2015 Anupam
24M2012  Anuj Yadav


# Taxi Fare Prediction

## Description
This project implements a neural network to predict taxi fares based on trip data such as pickup/dropoff locations, passenger count, and time-related features. 
The model is trained using the Adam optimization algorithm and evaluates performance using metrics such as RMSE and \( R^2 \) scores.

---

## Requirements
- Python 3.7+
- Required Python Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `geopy`

Install required libraries using:
```bash
pip install numpy pandas matplotlib geopy
```

---

## Dataset
- The model expects a CSV file with taxi trip data. 
- Example file: **`train.csv`** (should contain the following columns):
  - `key`: A unique identifier for each trip.
  - `pickup_datetime`: The timestamp of the pickup.
  - `pickup_longitude` & `pickup_latitude`: Coordinates of the pickup location.
  - `dropoff_longitude` & `dropoff_latitude`: Coordinates of the dropoff location.
  - `passenger_count`: Number of passengers in the trip.
  - `fare_amount`: Actual fare amount for the trip.

---

## Features
The model generates the following features from the data:
1. **Distance**: Geodesic distance between pickup and dropoff points.
2. **Hour and Weekday**: Extracted from `pickup_datetime`.
3. **Weekend**: Boolean feature indicating if the trip occurred on a weekend.
4. **Rush Hour**: Boolean feature indicating if the trip occurred during rush hours.
5. **Normalized Numerical Features**: Scaled versions of numeric inputs for better training.

---

## Model Architecture
- Input Layer: Number of neurons = Number of input features.
- Hidden Layers: Fully connected layers with configurable dimensions.
  - Default dimensions: `[64, 32, 21]`.
  - Activation Function: ReLU.
- Output Layer: A single neuron with a linear activation function.

---

## Code Structure

1. **`TaxiFarePrediction` Class**:
   - Implements a feedforward neural network.
   - Supports forward propagation, backpropagation, and weight updates using Adam optimizer.
   - Includes methods for training, evaluation, and metric plotting.

2. **`data_processing` Function**:
   - Reads and preprocesses the dataset.
   - Extracts features, filters outliers, and splits data into training and testing sets.

3. **Training Script**:
   - Configures model hyperparameters.
   - Trains the model for a specified number of epochs with mini-batch gradient descent.
   - Logs training and testing performance metrics for each epoch.

4. **Plotting Function**:
   - Generates and saves loss and RMSE trends as a PNG file.

---

## Usage

### 1. Prepare the Dataset
Ensure the dataset file (`train.csv`) is in the project directory and contains the necessary columns.

### 2. Run the Training Script
```python
# Import and process data
X_train, X_test, y_train, y_test = data_processing('train.csv')

# Model configuration
hidden_dims = [64, 32, 21]
input_dim = X_train.shape[1]
batch_size = 200
num_epochs = 50
optimizer = "adam"
optimizer_params = {
    'learning_rate': 0.01,
    'beta1': 0.85,
    'beta2': 0.999,
    'eps': 1e-8
}

# Initialize and train model
model = TaxiFarePrediction(input_dim, hidden_dims)
train_losses, test_losses, train_rmse, test_rmse = model.train(
    X_train, y_train, X_test, y_test, num_epochs, batch_size, optimizer, optimizer_params
)

# Plot metrics
model.plot_metrics(train_losses, test_losses, train_rmse, test_rmse)
```

### 3. Outputs
- Logs for each epoch, including:
  - Train/Test Loss
  - Train/Test RMSE
  - Train/Test \( R^2 \) scores
- A plot showing trends in loss and RMSE (`loss_and_rmse.png`).

---


## Customization

1. **Change Hidden Layer Dimensions**:
   Modify the `hidden_dims` list to adjust the network architecture.
   ```python
   hidden_dims = [512, 256, 128, 64]
   ```

2. **Adjust Hyperparameters**:
   Modify `optimizer_params` to change learning rate or Adam parameters.

3. **Increase Epochs**:
   Adjust `num_epochs` for longer training.

---

