import numpy as np
import pandas as pd

def haversine(lat1, lon1):
    # Radius of Earth in kilometers
    R = 6371.0
    lat2, lon2 = 40.785091, -73.968285

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

def create_new_columns(df):
    
    # Price Bins
    bin_edges = [0, 220000, 280000, 350000, 425000, 500000, 570000, 630000, 690000,
                750000, 800000, 900000, 1000000, 1200000, 1300000, 1500000, 1900000,
                2400000, 3500000, 7500000, float('inf')]

    bin_labels = ['0 to 220K', '220K to 280K', '280K to 350K', '350K to 425K', '425K to 500K',
                '500K to 570K', '570K to 630K', '630K to 690K', '690K to 750K',
                '750K to 800K', '800K to 900K', '900K to 1M', '1M to 1.20M',
                '1.20M to 1.3M', '1.3M to 1.5M', '1.5M to 1.9M', '1.9M to 2.4M',
                '2.4M to 3.5M', '3.5M to 7.5M', '7.5M and above']

    # Assign bins to data
    df['PRICE_BINS'] = pd.cut(df['PRICE'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    num_bins = 10

    bin_edges = [0, 800, 1100, 1400, 1800, 2165, 2180, 2195, 2210, 3360, float('inf')]

    bin_labels = [f'{bin_edges[i]}-{bin_edges[i+1]}' for i in range(num_bins)]

    # Assign bins to data
    df['PROPERTYSQFT_BINS'] = pd.cut(df['PROPERTYSQFT'], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Calculate distance to Central Park and add it to DataFrame
    df['DISTANCE'] = df.apply(lambda row: haversine(row['LATITUDE'], row['LONGITUDE']), axis=1)

    return df

def drop_columns(df):
    columns_to_drop = [
        'BROKERTITLE', 'TYPE', 'PRICE', 'PROPERTYSQFT', 'ADDRESS', 'STATE', 
        'MAIN_ADDRESS', 'ADMINISTRATIVE_AREA_LEVEL_2', 'LONG_NAME', 'FORMATTED_ADDRESS', 
        'LATITUDE', 'LONGITUDE', 'STREET_NAME'
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    return df

def load_data(train_data_path, train_label_path, test_data_path):
    train_df = pd.read_csv(train_data_path)
    y_train = pd.read_csv(train_label_path)

    train_df['BEDS'] = y_train['BEDS']

    # drop duplicates
    train_df = train_df.drop_duplicates()

    train_df = create_new_columns(train_df)
    train_df = drop_columns(train_df)

    y_train = train_df['BEDS']
    X_train = train_df.drop('BEDS', axis=1)

    X_test = pd.read_csv(test_data_path)

    X_test = create_new_columns(X_test)
    X_test = drop_columns(X_test)

    return X_train, y_train, X_test

def encode(X_train, X_test):

    # Combine training and test data
    combined_data = pd.concat([X_train, X_test], ignore_index=True)

    # Encode categorical variables
    combined_encoded = pd.get_dummies(combined_data, columns=['LOCALITY'
                                                              ,'SUBLOCALITY'
                                                              ,'PRICE_BINS'
                                                              ,'PROPERTYSQFT_BINS'
                                                        ])

    # Split back into training and test data
    X_train_encoded = combined_encoded.iloc[:len(X_train)]
    X_test_encoded = combined_encoded.iloc[len(X_train):]

    X_train = X_train_encoded
    X_test = X_test_encoded

    return X_train, X_test

def scaler(X_train, X_test):
    mean = X_train.mean() + 1e-9
    std_dev = X_train.std() + 1e-9

    # transform data
    X_train = (X_train - mean) / std_dev
    X_test = (X_test - mean) / std_dev

    X_train = X_train.values
    X_test = X_test.values

    return X_train, X_test

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def sigmoid(x):
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

class CustomMLP:
    def __init__(self, n_inputs, n_outputs, hidden_layers, activation='relu', alpha = 0.0001):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.alpha = alpha
        
        # Select activation and derivative functions
        if self.activation == 'relu':
            self.activation_function = relu
            self.activation_derivative = relu_derivative
        elif self.activation == 'sigmoid':
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function specified. Use 'relu' or 'sigmoid'")
        
        layers = [self.n_inputs] + self.hidden_layers + [self.n_outputs]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layers)-1):

            self.weights.append(np.random.uniform(low=-0.1, high=0.1, size=(layers[i], layers[i+1])))

            self.biases.append(np.zeros((1, layers[i+1])))
    
    def feedforward(self, X):
        self.activations = [X]
        self.z_values = []  # pre-activation values
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                activation = self.activation_function(z)
            else:
                activation = softmax(z)
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backprop(self, learning_rate, X, y):
        y = to_categorical(y, self.n_outputs)
        delta = cross_entropy(self.activations[-1], y)
        for i in reversed(range(len(self.weights))):
            z = self.z_values[i]
            if i < len(self.weights) - 1:
                delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(z)

            # Regularization term
            l2_regularization = self.alpha * self.weights[i]
            
            self.weights[i] -= learning_rate * (np.dot(self.activations[i].T, delta) + l2_regularization)
            self.biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)
    
    def train(self, X, y, learning_rate, n_epochs, batch_size):
        for epoch in range(n_epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
                self.feedforward(X_batch)
                self.backprop(learning_rate, X_batch, y_batch)
            
    
    def compute_loss(self, X, y):
        y_pred = self.feedforward(X)
        y_true = to_categorical(y, self.n_outputs)
        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
        return loss
    
    def accuracy(self, X, y_true):
        y_pred = self.feedforward(X)
        correct_predictions = np.argmax(y_pred, axis=1) == y_true
        return np.mean(correct_predictions)
    
    def predict(self, X):
        y_pred = self.feedforward(X)
        return np.argmax(y_pred, axis=1)

# main
train_data_path = 'train_data.csv'
train_label_path = 'train_label.csv'
test_data_path = 'test_data.csv'
test_label_path = 'test_label.csv'

X_train, y_train, X_test = load_data(train_data_path, train_label_path, test_data_path)

X_train, X_test = encode(X_train, X_test)

X_train, X_test = scaler(X_train, X_test)

n_inputs = X_train.shape[1]
n_outputs = 51

hidden_layers = [50]
learning_rate = 0.03
n_epochs = 1000
batch_size = 500
activation = 'relu'
alpha = 0.01 

# Initialize the MLP
mlp = CustomMLP(n_inputs, n_outputs, hidden_layers, activation = activation, alpha=alpha)

# Train the MLP
mlp.train(X_train, y_train, learning_rate, n_epochs, batch_size)

y_pred = mlp.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=['BEDS'])
y_pred_df.to_csv('output.csv', index=False)
