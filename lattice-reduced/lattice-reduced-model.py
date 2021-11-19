import tensorflow as tf

import numpy as np
import pandas as pd

import tensorflow_lattice as tfl

def reduce_x(df_in):
    reduced_df = pd.DataFrame()
    reduced_df['avgsim0'] = df_in.iloc[:, 0:31].mean(axis=1)
    return reduced_df

LEARNING_RATE = 0.01
BATCH_SIZE = 30
NUM_EPOCHS = 100

train = pd.read_csv('training-sim-scores.csv')
val = pd.read_csv('validation-sim-scores.csv')
test = pd.read_csv('test-sim-scores.csv')

X_train = train.drop(['reference', 'degraded'], axis=1)
X_val = val.drop(['reference', 'degraded'], axis=1)
X_test = test.drop(['reference', 'degraded'], axis=1)

Y_train = X_train.pop('moslqs')
Y_val = X_val.pop('moslqs')
Y_test = X_test.pop('moslqs')

reduced_x_train = reduce_x(X_train)
reduced_x_val = reduce_x(X_val)

lattice_sizes = [3]


combined_calibrators = tfl.layers.ParallelCombination()

calibrator = tfl.layers.PWLCalibration(
    input_keypoints=np.linspace(
        reduced_x_train['avgsim0'].min(), reduced_x_val['avgsim0'].max(), num=20),
        output_min=1.0,
        output_max=lattice_sizes[0] - 1.0,
        monotonicity='increasing',
        kernel_regularizer=('wrinkle',0.0,0.2)
)
combined_calibrators.append(calibrator)

lattice = tfl.layers.Lattice(
    lattice_sizes=lattice_sizes,
    output_min=1.0,
    output_max=5.0
)

model = tf.keras.models.Sequential()
model.add(combined_calibrators)
model.add(lattice)

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    metrics=['mse', 'mae'])
model.fit(
    reduced_x_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=(reduced_x_val, Y_val),
    verbose=2)

model.evaluate(reduced_x_val, Y_val)
predictions = model.predict(reduced_x_val)
df = pd.DataFrame(data=predictions, columns=['moslqo'])
df.to_csv('./valresults.csv', index=0)

model.save('saved_model/model')





