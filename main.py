#importing basic library
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from geopy.distance import geodesic

# calculating relu and its dervative
# Returns the ReLU value of the input x
def relu(x):
    return np.maximum(0, x)

# Returns the derivative of the ReLU value of the input x
def relu_derivative(x):
    return (x>0).astype(int)
def linear(x):
  return x
def mean_squared_error(y_true, y_pred):
        """Mean Squared Error loss."""
        return np.mean((y_true - y_pred) ** 2)
def mean_squared_error_derivative(y_true, y_pred):

        return -1*(y_true - y_pred)

class TaxiFarePrediction :
  def __init__(self,input_dim,hidden_dims):
    #Initialize weights and biases for all hidden and output layers
    self.weights = []
    self.biases = []

    for i in range(len(hidden_dims)):
          if(i==0):
              self.weights.append(np.random.randn(hidden_dims[i],input_dim)*(1/np.sqrt(input_dim)))
              self.biases.append(np.random.randn(hidden_dims[i],1).T*(1/np.sqrt(input_dim)))
          else:
              self.weights.append(np.random.randn(hidden_dims[i],hidden_dims[i-1])*(1/np.sqrt(hidden_dims[i-1])))
              self.biases.append(np.random.randn(1,hidden_dims[i])*(1/np.sqrt(hidden_dims[i-1])))



    self.weights.append(np.random.randn(1,hidden_dims[-1])*(1/np.sqrt(hidden_dims[-1])))
    #print(self.weights)
    self.biases.append(np.random.randn(1,1)*(1/np.sqrt(hidden_dims[-1])))
    #print(self.biases)

  def forward(self, X):
        self.weighted_sum=[]
        self.actived_value=[]
        '''
        Parameters
        X : input data, numpy array of shape (N, D) where N is the number of examples and D
            is the dimension of each example
         '''
        # Forward pass
        # Compute activations for all the nodes with the corresponding
        #activation function of each layer applied to the hidden nodes

        output_values=[]
        layers=len(self.weights)
        for layer in range(layers):
            if layer==0:
              self.weighted_sum.append(np.dot(X,self.weights[layer].T)+self.biases[layer])
              self.actived_value.append(relu(self.weighted_sum[layer]))

            elif layer==(layers-1):
              self.weighted_sum.append(np.dot(self.actived_value[layer-1],self.weights[layer].T)+self.biases[layer])
              self.actived_value.append(linear(self.weighted_sum[layer]))

            else:
                self.weighted_sum.append(np.dot(self.actived_value[layer-1],self.weights[layer].T)+self.biases[layer])
                self.actived_value.append(relu(self.weighted_sum[layer]))




        # Calculate the output probabilities of shape (N, 1) where N is number of examples
        output_values=self.actived_value[-1]
        return output_values

  def backward(self,X,y):
    num_layers=len(self.weights)

    self.grad_weights = [None]*num_layers
    self.grad_biases = [None]*num_layers


    y_pred=self.actived_value[-1]
    loss_val=mean_squared_error(y,y_pred)
    #print(loss_val)

    dloss_dz=[None]*num_layers
    dloss_da=[None]*num_layers


    dloss_da[-1]=mean_squared_error_derivative(y,y_pred)
    dloss_dz[-1]=dloss_da[-1]

    #print(dloss_da[-1])


    self.grad_weights[-1]=np.dot(self.actived_value[-2].T,dloss_dz[-1]).T
    self.grad_biases[-1]=np.sum(dloss_dz[-1],axis=0)

    #print(self.grad_weights[-1].shape)
    for layer in reversed(range(num_layers-1)):
            if layer==0:
                dloss_dz[layer]=np.dot(dloss_dz[layer+1],self.weights[layer+1])*relu_derivative(self.weighted_sum[layer])
                self.grad_weights[layer]=np.dot(X.T,dloss_dz[layer]).T
                self.grad_biases[layer]=np.sum(dloss_dz[layer],axis=0)

            else:
                dloss_dz[layer]=np.dot(dloss_dz[layer+1],self.weights[layer+1])*relu_derivative(self.weighted_sum[layer])
                self.grad_weights[layer]=np.dot(self.actived_value[layer-1].T,dloss_dz[layer]).T
                self.grad_biases[layer]=np.sum(dloss_dz[layer],axis=0)


    return self.grad_weights, self.grad_biases

  def step_adam(self, weights, biases, delta_weights, delta_biases, optimizer_params,epochs):
        layers=len(weights)
        updated_W=[]
        updated_B=[]
        epochs=epochs+1
        '''
       
        '''
        learning_rate = optimizer_params['learning_rate']
        beta = optimizer_params['beta1']
        gamma = optimizer_params['beta2']
        eps = optimizer_params['eps']
        def s_hat(x):
            return x/(1-gamma**epochs)
        def v_hat(x):
            return x/(1-beta**epochs)



        if epochs==1:
            self.v_tw=[]
            self.s_tw=[]

            self.s_tb=[]
            self.v_tb=[]

            for layer in range(layers):

                self.v_tw.append(np.zeros(delta_weights[layer].shape))
                self.s_tw.append(np.zeros(delta_weights[layer].shape))

                self.v_tb.append(np.zeros(delta_biases[layer].shape))
                self.s_tb.append(np.zeros(delta_biases[layer].shape))

                self.v_tw[layer]=beta*self.v_tw[layer]+(1-beta)*delta_weights[layer]
                self.s_tw[layer]=gamma*self.s_tw[layer]+(1-gamma)*(delta_weights[layer]*delta_weights[layer])

                self.v_tb[layer]=beta*self.v_tb[layer]+(1-beta)*delta_biases[layer]
                self.s_tb[layer]=gamma*self.s_tb[layer]+(1-gamma)*(delta_biases[layer]*delta_biases[layer])

                v_hat_w=v_hat(self.v_tw[layer])
                s_hat_w=s_hat(self.s_tw[layer])

                v_hat_b=v_hat(self.v_tb[layer])
                s_hat_b=s_hat(self.s_tb[layer])


                updated_w_layer = weights[layer] - (learning_rate * v_hat_w) / (np.sqrt(s_hat_w) + eps)
                updated_b_layer = biases[layer] - (learning_rate * v_hat_b) / (np.sqrt(s_hat_b) + eps)



                updated_W.append(updated_w_layer)
                updated_B.append(updated_b_layer)

        else:


            for layer in range(layers):



                self.v_tw[layer]=beta*self.v_tw[layer]+(1-beta)*delta_weights[layer]
                self.s_tw[layer]=gamma*self.s_tw[layer]+(1-gamma)*(delta_weights[layer]*delta_weights[layer])

                self.v_tb[layer]=beta*self.v_tb[layer]+(1-beta)*delta_biases[layer]
                self.s_tb[layer]=gamma*self.s_tb[layer]+(1-gamma)*(delta_biases[layer]*delta_biases[layer])

                v_hat_w=v_hat(self.v_tw[layer])
                s_hat_w=s_hat(self.s_tw[layer])

                v_hat_b=v_hat(self.v_tb[layer])
                s_hat_b=s_hat(self.s_tb[layer])

                updated_w_layer = weights[layer] - (learning_rate * v_hat_w) / (np.sqrt(s_hat_w) + eps)
                updated_b_layer = biases[layer] - (learning_rate * v_hat_b) / (np.sqrt(s_hat_b) + eps)



                updated_W.append(updated_w_layer)
                updated_B.append(updated_b_layer)










        #Return updated weights and biases for the hidden layer based on the update rules for Adam Optimizer

        return updated_W, updated_B




  def train(self, X_train, y_train, X_eval, y_eval, num_epochs, batch_size, optimizer, optimizer_params):
    train_losses = []
    test_losses = []
    train_mae = []
    test_mae = []

    for epoch in range(num_epochs):
        # Divide X, y into batches
        X_batches = np.array_split(X_train, X_train.shape[0] // batch_size)
        y_batches = np.array_split(y_train, y_train.shape[0] // batch_size)

        for X, y in zip(X_batches, y_batches):
            # Forward pass
            self.forward(X)
            # Backpropagation and gradient descent weight updates
            dW, db = self.backward(X, y)
            if optimizer == "adam":
                self.weights, self.biases = self.step_adam(
                    self.weights, self.biases, dW, db, optimizer_params, epoch
                )


        # Compute training loss and RMSE
        train_preds = self.forward(X_train)
        train_loss = np.mean((y_train - train_preds) ** 2)
        train_mae_val = np.mean(np.abs(y_train - train_preds))  # RMSE for training
        train_mae.append(train_mae_val)

        # R² for training
        ss_res_train = np.sum((y_train - train_preds) ** 2)
        ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
        r2_score_train = 1 - (ss_res_train / ss_tot_train)

        # Log training metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train RMSE: {train_mae_val:.4f}, Train R²: {r2_score_train:.4f}")
        train_losses.append(train_loss)

        # Compute test loss and RMSE
        test_preds = self.forward(X_eval)
        test_loss = np.mean((y_eval - test_preds) ** 2)
        test_mae_val = np.mean(np.abs(y_eval - test_preds))  # RMSE for testing
        test_mae.append(test_mae_val)

        # R² for testing
        ss_res_test = np.sum((y_eval - test_preds) ** 2)
        ss_tot_test = np.sum((y_eval - np.mean(y_eval)) ** 2)
        r2_score_test = 1 - (ss_res_test / ss_tot_test)

        # Log testing metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae_val:.4f}, Test R²: {r2_score_test:.4f}")
        test_losses.append(test_loss)

    return train_losses, test_losses, train_mae, test_mae






  def plot_metrics(self, train_losses, test_losses, train_mae, test_mae):
    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # RMSE Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_mae, label='Train MAE')
    plt.plot(test_mae, label='Test MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_and_mae.png')

#Preprocessing
def data_processing(dataset):
       file_path = dataset  # Replace with your file path
       data = pd.read_csv(file_path)
       data = data.drop(columns=['key'])
       data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], errors='coerce')
       data['hour'] = data['pickup_datetime'].dt.hour
       data['weekday'] = data['pickup_datetime'].dt.weekday
       data = data.drop(columns=['pickup_datetime'])
       data = data[(data['fare_amount'] > 0) & (data['fare_amount'] <= 500)]
       data = data[(data['passenger_count'] > 0) & (data['passenger_count'] <= 6)]
       data = data[(data['pickup_longitude'] >= -75) & (data['pickup_longitude'] <= -72)]
       data = data[(data['dropoff_longitude'] >= -75) & (data['dropoff_longitude'] <= -72)]
       data = data[(data['pickup_latitude'] >= 40) & (data['pickup_latitude'] <= 42)]
       data = data[(data['dropoff_latitude'] >= 40) & (data['dropoff_latitude'] <= 42)]
       data['distance'] = data.apply(
           lambda row: geodesic(
               (row['pickup_latitude'], row['pickup_longitude']),
               (row['dropoff_latitude'], row['dropoff_longitude'])
           ).km, axis=1
       )
       data = data[data['distance'] > 0]
       data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
       data['rush_hour'] = data['hour'].isin(list(range(7, 10)) + list(range(16, 19))).astype(int)

       numerical_features = [
           'pickup_longitude', 'pickup_latitude',
           'dropoff_longitude', 'dropoff_latitude',
           'passenger_count', 'hour', 'weekday', 'distance'
       ]
       data[numerical_features] = (data[numerical_features] - data[numerical_features].mean()) / data[numerical_features].std()

       X = data.drop(columns=['fare_amount']).values
       y = data['fare_amount'].values.reshape(-1, 1)
       split_index = int(0.8 * len(data))
       X_train, X_test = X[:split_index], X[split_index:]
       y_train, y_test = y[:split_index], y[split_index:]

       return X_train,X_test,y_train,y_test

dataset='C:\\Users\\prabhat patel\\OneDrive\\Desktop\\fml project\\train.csv\\train.csv'
X_train,X_test,y_train,y_test=data_processing(dataset)


optimizer = "adam"
optimizer_params = {
     'learning_rate': 0.01,
    'beta1' : 0.85,
    'beta2' : 0.999,
     'eps' : 1e-8
  }

hidden_dims = [64,32,21]
input_dim = X_train.shape[1]
batch_size=200
num_epochs=25

model=TaxiFarePrediction(input_dim,hidden_dims)

train_losses, test_losses , train_mae, test_mae = model.train(X_train, y_train, X_test, y_test,num_epochs, batch_size, optimizer, optimizer_params)

model.plot_metrics(train_losses, test_losses, train_mae, test_mae)
