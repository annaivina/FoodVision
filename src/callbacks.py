import tensorflow as tf
import os
import datetime 

def get_callbacks(experiment_name="", model_name="", is_fine_tune=False):
    
    os.makedirs(experiment_name, exist_ok=True)

    log_dir = os.path.join(experiment_name, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_dir = os.path.join(experiment_name, "checkpoints", model_name )
   
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


    #Tensorboard - there is no more tensorboard. One can try just save the logs
    csvlogger_callback = tf.keras.callbacks.CSVLogger(filename=os.path.join(log_dir,"training.csv"), separator='.', append=False)
    #Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #Saving best model 
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "best_model.keras"),
                                                             save_best_only=True,
                                                             monitor="val_loss")
    

    return [csvlogger_callback, early_stopping, checkpoint_callback] if is_fine_tune else [csvlogger_callback]

  





