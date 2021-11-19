import tensorflow as tf
import pandas as pd

test = pd.read_csv('test-sim-scores.csv')

X_test = test.drop(['reference', 'degraded'], axis=1)

Y_test = X_test.pop('moslqs')

def reduce_x(df_in):
    reduced_df = pd.DataFrame()
    reduced_df['avgsim0'] = df_in.iloc[:, 0:31].mean(axis=1)
    return reduced_df

test_xs = reduce_x(X_test)

model = tf.keras.models.load_model('saved_model/model')

print(model.evaluate(test_xs, Y_test))

predictions = model.predict(test_xs)
df = pd.DataFrame(data=predictions, columns=['moslqo'])
df.to_csv('./lattice_reduced_lqo_results.csv', index=0)