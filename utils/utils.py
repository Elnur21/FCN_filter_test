import numpy as np
import pandas as pd
import matplotlib
from tensorflow.keras.losses import categorical_crossentropy

matplotlib.use('agg')
import matplotlib.pyplot as plt


from utils.constants import ARCHIVE_NAMES  as ARCHIVE_NAMES

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def save_logs(output_directory, hist, y_pred, y_true, duration,val_acc, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float64), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    # df_best_model['best_model_val_loss'] = val_loss
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = val_acc
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    # plot_epochs_metric(hist.history, output_directory + 'epochs_loss.png')

    return df_metrics

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_filters(model):
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name]

    for layer in conv_layers:
    # Get the weights of the layer
        filters, biases = layer.get_weights()
        
        # Normalize the filters
        filters = (filters - np.min(filters)) / (np.max(filters) - np.min(filters))
        
        filters_reshaped = filters.reshape(-1, np.prod(filters.shape[1:]))
        # Determine the number of filters in the layer
        n_filters = filters.shape[-1]
        
        for i in range(n_filters):
            plt.scatter(range(len(filters_reshaped[i])), filters_reshaped[i], s=5)
        # Plot each filter
        for i in range(n_filters):
            # Get the ith filter
            filter_i = filters[:, :, i]  # Assuming 2D filters, adjust if needed
            
            # Plot the filter
            plt.figure()
            plt.scatter(filter_i)
            plt.title(f'Filter {i+1}')
            plt.savefig(f'filter_{i+1}.png')  # Save the figure

def custom_evaluate(model, x_test, y_test):
    # Predict the probabilities for each class
    y_pred_probs = model.predict(x_test)
    
    # Convert probabilities to class labels
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    # Convert predicted labels to integers
    y_pred_labels = y_pred_labels.astype(int)

    # Calculate cross-entropy loss
    loss = -np.mean(np.log(y_pred_probs[np.arange(len(y_test)), y_pred_labels]))
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_labels == y_test)
    
    return loss, accuracy 
    

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float64), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res