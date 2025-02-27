import tensorflow as tf
import os
import datetime 

def get_callbacks(experiment_name="", model_name="", is_fine_tune=False):
    
    os.makedirs(experiment_name, exist_ok=True)

    log_dir = os.path.join(experiment_name, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(experiment_name, "checkpoints", model_name )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    #Tensorboard 
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #Saving best model 
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "best_model.keras"),
                                                             save_best_only=True,
                                                             monitor="val_loss")
    

    return [tensorboard_callback, early_stopping, checkpoint_callback] if is_fine_tune else [tensorboard_callback]

  





