import warnings
import numpy as np


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import classification_report
from utils import CGAN, OracleFCN, WTST, imbalance, write_to_csv, oversample, no_oversample

warnings.filterwarnings('ignore')


data = np.load('datasets/tor_data.npy')
labels = np.load('datasets/tor_labels.npy')
dataset_name = 'tor'


def nn(x, y, x_test, y_test):

    model = Sequential()
    model.add(Dense(10, input_shape=(x.shape[-1],), activation='relu'))
    # model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs=20, batch_size=32, verbose=0, validation_split=0.2)

    y_pred = model.predict(x_test)
    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.
    result = classification_report(y_test, y_pred, output_dict=True)

    maj_f1 = result['0.0']['f1-score']
    min_f1 = result['1.0']['f1-score']

    return maj_f1, min_f1

# Running All the tests

oversample(
              dataset_name=dataset_name,
              data = data,
              labels = labels,
              wtst_param_number_of_accepted_failed_attempts = 15,
              wtst_param_epoch_unit = 100,
              gan_param_number_of_generated_samples_perclass = 500,
              classifier = nn,
              fcn_number_of_hidden_layers_for_classifier = 2,
              fcn_number_of_neurons_in_layer_for_classifier = 100,
              fcn_number_of_epochs_for_training_classifier = 100,
               
              cas_number_of_hidden_layers_for_classifier = 2,
              cas_number_of_neurons_in_layer_for_classifier = 20,
              cas_number_of_epochs_for_training_classifier = 20,
              
              mis_number_of_hidden_layers_for_classifier = 2,
              mis_number_of_neurons_in_layer_for_classifier = 100,
              mis_number_of_epochs_for_training_classifier = 100,
               
              oracle_param_number_of_epochs_for_training_feature_extractor = 200,
              no_oracle_training_epochs = 1500,
              maj_counts = [200, 500, 1000],
              im_ratios = [0.1, 0.2, 0.3, 0.4]
              )

"""##No oversamping"""

no_oversample(
              dataset_name = dataset_name,
              data = data,
              labels = labels,
              classifier = nn,
)