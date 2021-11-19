import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import pandas as pd

train = pd.read_csv('training-sim-scores.csv')
val = pd.read_csv('validation-sim-scores.csv')
test = pd.read_csv('test-sim-scores.csv')

X_train = train.drop(['reference', 'degraded'], axis=1)
X_val = val.drop(['reference', 'degraded'], axis=1)
X_test = test.drop(['reference', 'degraded'], axis=1)

Y_train = X_train.pop('moslqs')
Y_val = X_val.pop('moslqs')
Y_test = X_test.pop('moslqs')

print(X_train)
model = Sequential()
model.add(Dense(35, input_dim=32, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dropout(input_dim=20, rate=0.2))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, Y_train, epochs=500, batch_size=31, verbose=1, validation_data=(X_val, Y_val))

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


model.save_weights('./models/model1')