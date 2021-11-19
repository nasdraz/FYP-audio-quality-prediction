import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import tensorflow_lattice as tfl


test = pd.read_csv('test-sim-scores.csv')

X_test = test.drop(['reference', 'degraded'], axis=1)

Y_test = X_test.pop('moslqs')

feature_names = ['fvnsim0', 'fvnsim1', 'fvnsim2', 'fvnsim3', 'fvnsim4', 'fvnsim5',
                 'fvnsim6', 'fvnsim7', 'fvnsim8', 'fvnsim9', 'fvnsim10', 'fvnsim11',
                 'fvnsim12', 'fvnsim13', 'fvnsim14', 'fvnsim15', 'fvnsim16', 'fvnsim17',
                 'fvnsim18', 'fvnsim19', 'fvnsim20', 'fvnsim21', 'fvnsim22', 'fvnsim23',
                 'fvnsim24', 'fvnsim25', 'fvnsim26', 'fvnsim27', 'fvnsim28', 'fvnsim29',
                 'fvnsim30', 'fvnsim31']

def extract_features(dataframe, feature_names=feature_names):
  features = []
  for feature_name in feature_names:
      features.append(dataframe[feature_name].values.astype(float))
  return features

test_xs = extract_features(X_test)

model = tf.keras.models.load_model('saved_model/model')

print(model.evaluate(test_xs, Y_test))

predictions = model.predict(test_xs)
df = pd.DataFrame(data=predictions, columns=['moslqo'])
df.to_csv('./lattice_lqo_results.csv', index=0)



