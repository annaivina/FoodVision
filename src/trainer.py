import tensorflow as tf

class Trainer:

    def __init__(self, model, train_data, test_data, epochs=10, callbacks=None):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        self.callbacks = callbacks if callbacks else []

    
    def train(self):
        history = self.model.fit(
            self.train_data,
            steps_per_epoch=len(self.train_data),
            validation_data=self.test_data,
            validation_steps=int(0.15 * len(self.test_data)),
            epochs=self.epochs,
            callbacks = self.callbacks
        )

        return history 
    
    def evaluate(self):

        results = self.model. evaluate(self.test_data)
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")
        return results


    



 
