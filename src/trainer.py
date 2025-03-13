import tensorflow as tf
from tqdm import tqdm 

class Trainer:

    def __init__(self, build_model, train_data, test_data, epochs=10, callbacks=None):
        self.build_model = build_model
        self.model = build_model.model
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.callbacks = callbacks if callbacks else []
        
        self.model_optimizer = self.model.optimizer
        self.model_metric = self.model.metrics[0]
        self.model_loss = tf.keras.losses.get(self.model.loss)
   

    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch, training=True)
            loss_train = tf.reduce_mean(self.model_loss(y_batch, y_pred))

        part_der = tape.gradient(loss_train, self.model.trainable_weights)
        self.model_optimizer.apply_gradients(zip(part_der, self.model.trainable_weights))
        self.model_metric.update_state(y_batch, tf.argmax(y_pred, axis=-1))

        return loss_train
    

    @tf.function
    def validation_step(self, x_batch_val, y_batch_val):
        y_pred_val = self.model(x_batch_val, training=False)
        loss_val = tf.reduce_mean(self.model_loss(y_batch_val, y_pred_val))
        self.model_metric.update_state(y_batch_val, tf.argmax(y_pred_val, axis=-1)) # Take the max value form the predicitons 

        return loss_val

    def train(self):
        history = {"train_loss": [], "train_accuracy": [], "valid_loss": [], "valid_accuracy": []}
    
        for epoch in range(self.epochs):
            self.model_metric.reset_state()  # Reset metric before each epoch 

            epoch_loss_train_sum = 0
            num_batches = 0

            progress_bar = tqdm(self.train_data, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")

            for (x_batch, y_batch) in self.train_data:
                loss_train = self.train_step(x_batch, y_batch)

                epoch_loss_train_sum +=loss_train.numpy()
                num_batches +=1
                #Monitor the progress 
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss_train.numpy(), acc=self.model_metric.result().numpy())
            
            progress_bar.close()

            epoch_loss_train = epoch_loss_train_sum / num_batches
            train_acc = self.model_metric.result().numpy()


            history['train_loss'].append(epoch_loss_train)
            history['train_accuracy'].append(train_acc)

            print(f"\nEpoch {epoch + 1}/{self.epochs}: Train Loss: {epoch_loss_train:.4f}, Train Accuracy: {train_acc:.4f}")
            
            #Recet the result because we need to check the validation accuracy
            self.model_metric.reset_state()
            val_loss_sum = 0
            num_val_batches = 0
            val_progress_bar = tqdm(self.test_data.take(int(0.15 * len(self.test_data))), desc="Validating", unit="batch")

            for (x_batch_val, y_batch_val) in self.test_data.take(int(0.15 * len(self.test_data))):
                loss_val = self.validation_step(x_batch_val, y_batch_val)
                
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(val_loss=loss_val.numpy(), val_acc=self.model_metric.result().numpy())

                val_loss_sum +=loss_val.numpy()
                num_val_batches+=1

            val_progress_bar.close()

            epoch_val_loss = val_loss_sum / num_val_batches
            valid_acc = self.model_metric.result().numpy()
            print(f"{epoch} out of {self.epochs}: Valid Loss: {epoch_val_loss:.4f} Valid accuracy: {valid_acc:.4f}") 
            history['valid_loss'].append(epoch_val_loss)
            history['valid_accuracy'].append(valid_acc)  
        
        return history








    
    def evaluate(self):

        results = self.model. evaluate(self.test_data)
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")
        return results


    



 
