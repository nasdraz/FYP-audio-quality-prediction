from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

model = Sequential()
model.add(Dense(35, input_dim=32, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

model.load_weights('./models/model1')
predictions = model.predict(X_test)
print(predictions)
df = pd.DataFrame(data=predictions, columns=['moslqo'])
df.to_csv('./test_results.csv', index=0)