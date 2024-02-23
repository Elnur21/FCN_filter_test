import time

from aeon.classification.deep_learning._fcn import FCNClassifier
from tensorflow import keras 

from utils.helper import *
from utils.constants import *
from utils.utils import *

def fit_classifier(df, output_directory):
    x_train=df[0]
    y_train=df[1]
    x_val=df[2]
    y_val=df[3]

    mini_batch_size = int(min(x_train.shape[0]/10, 16))

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    file_path = output_directory+'best_model.hdf5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

    callbacks = [reduce_lr,model_checkpoint]

    start_time = time.time() 
    fcn_model = FCNClassifier(batch_size=mini_batch_size, n_epochs=2000,
        verbose=True, callbacks=callbacks)

    hist = fcn_model.fit(x_train, y_train)
    duration = time.time() - start_time
    fcn_model.save_last_model(output_directory + 'last_model.hdf5')
    # fcn_model.save(output_directory+'last_model.hdf5')
    model = keras.models.load_model(output_directory+'best_model.hdf5')
    y_pred = model.predict(x_val)
    # convert the predicted from binary to integer 
    y_pred = np.argmax(y_pred , axis=1)
    save_logs(output_directory, hist, y_pred, y_val, duration)

for dataset_name in np.array(UNIVARIATE_DATASET_NAMES_2018)[:1]:
    output_directory = 'results_train2/' + '/' + dataset_name + '/'
    test_dir_df_metrics = output_directory + 'df_metrics.csv'
    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        create_directory(output_directory)
        df = read_dataset(dataset_name)
        fit_classifier(df,  output_directory)
        print('DONE')
        # the creation of this directory means
        create_directory(output_directory + '/DONE')

    