import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import tensorflow_lattice as tfl

LEARNING_RATE = 0.001
BATCH_SIZE = 30
NUM_EPOCHS = 500


train = pd.read_csv('training-sim-scores.csv')
val = pd.read_csv('validation-sim-scores.csv')
test = pd.read_csv('test-sim-scores.csv')

X_train = train.drop(['reference', 'degraded'], axis=1)
X_val = val.drop(['reference', 'degraded'], axis=1)
X_test = test.drop(['reference', 'degraded'], axis=1)

Y_train = X_train.pop('moslqs')
Y_val = X_val.pop('moslqs')
Y_test = X_test.pop('moslqs')

# Let's define our label minimum and maximum.
min_label, max_label = 0, 5
# Our lattice models may have predictions above 1.0 due to numerical errors.
# We can subtract this small epsilon value from our output_max to make sure we
# do not predict values outside of our label bound.
numerical_error_epsilon = 1e-5

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

def compute_quantiles(features,
                      num_keypoints=10,
                      clip_min=None,
                      clip_max=None,
                      missing_value=None):
  # Clip min and max if desired.
  if clip_min is not None:
    features = np.maximum(features, clip_min)
    features = np.append(features, clip_min)
  if clip_max is not None:
    features = np.minimum(features, clip_max)
    features = np.append(features, clip_max)
  # Make features unique.
  unique_features = np.unique(features)
  # Remove missing values if specified.
  if missing_value is not None:
    unique_features = np.delete(unique_features,
                                np.where(unique_features == missing_value))
  # Compute and return quantiles over unique non-missing feature values.
  return np.quantile(
      unique_features,
      np.linspace(0., 1., num=num_keypoints),
      interpolation='nearest').astype(float)

train_xs = extract_features(X_train)
val_xs = extract_features(X_val)


feature_configs = []

for i in range(32):
    feature_configs.append(tfl.configs.FeatureConfig(
        name='fvnsim'+ str(i),
        lattice_size=3,
        monotonicity='increasing',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X_train['fvnsim'+ str(i)],
            num_keypoints=5),
        # Per feature regularization.
        regularizer_configs=[
            tfl.configs.RegularizerConfig(name='calib_wrinkle', l2=0.1),
        ],
    ))

explicit_ensemble_model_config = tfl.configs.CalibratedLatticeEnsembleConfig(
    feature_configs=feature_configs,
    lattices=[['fvnsim0', 'fvnsim1', 'fvnsim2', 'fvnsim3'], ['fvnsim4', 'fvnsim5', 'fvnsim6', 'fvnsim7'],
              ['fvnsim8', 'fvnsim9', 'fvnsim10', 'fvnsim11'], ['fvnsim12', 'fvnsim13', 'fvnsim14', 'fvnsim15'],
              ['fvnsim16', 'fvnsim17', 'fvnsim18', 'fvnsim19'], ['fvnsim20', 'fvnsim21', 'fvnsim22', 'fvnsim23'],
              ['fvnsim24', 'fvnsim25', 'fvnsim26', 'fvnsim27'], ['fvnsim28', 'fvnsim29', 'fvnsim30', 'fvnsim31']],
    num_lattices=8,
    lattice_rank=4,
    output_min=min_label,
    output_max=max_label - numerical_error_epsilon,
    output_initialization=[min_label, max_label])

explicit_ensemble_model = tfl.premade.CalibratedLatticeEnsemble(
    explicit_ensemble_model_config)



explicit_ensemble_model.compile(
    loss='mse',
    metrics=['mse','mae'],
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
history = explicit_ensemble_model.fit(
    train_xs, Y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=True, validation_data=(val_xs, Y_val))
print('Test Set Evaluation...')
print(explicit_ensemble_model.evaluate(val_xs, Y_val))

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions = explicit_ensemble_model.predict(val_xs)
df = pd.DataFrame(data=predictions, columns=['moslqo'])
df.to_csv('./valresultscheck.csv', index=0)


#explicit_ensemble_model.save_weights('./saved_model_weights/model')
explicit_ensemble_model.save('saved_model/modelcheck')
