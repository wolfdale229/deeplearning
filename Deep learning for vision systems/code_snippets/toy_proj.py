import numpy as np
from sklearn.datasets import make_blobs
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



X, y = make_blobs(n_samples=1000, n_features=2, 
	centers=3, cluster_std=2, random_state=2)

# one_hot_encode y
y = to_categorical(y, num_classes=len(np.unique(y)))

# split dataset
split_size = 800

X_train, X_test = X[:split_size, :], X[split_size:, :]
y_train, y_test = y[:split_size], y[split_size:]

# build model
model = Sequential()

model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy',
    metrics=['accuracy'], optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000,
	verbose=1)