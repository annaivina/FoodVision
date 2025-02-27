import tensorflow as tf
from tensorflow.keras import mixed_precision
import os 


class ModelBuilder:
    def __init__(self, input_shape = (224,224,3), num_classes=2, activation='softmax', lr=0.001, mixed_precision_training=False):

        if mixed_precision_training:
            mixed_precision.set_global_policy(policy='mixed_float16')

        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.lr = lr
        self.base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=self.input_shape)
        self.model = self._build_model()
        self.compile_model()


    def _build_model(self):

        self.base_model.trainable = False
        inputs = tf.keras.Input(shape=self.input_shape, name='input_layer')
        x = self.base_model(inputs, training=False)  # Use base model in inference mode
        x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling_layer')(x)
        x = tf.keras.layers.Dense(self.num_classes, name='dense_output_layer')(x)
        outputs = tf.keras.layers.Activation(self.activation, dtype=tf.float32, name='softmax_output')(x)  # Ensure dtype is float32

        model = tf.keras.Model(inputs, outputs)
        
        return model
    

    def fine_tune(self, trainable_layers=5, fine_tune_lr=0.0001):
        """
            Please use the learning rate for fine tune. Default will be at least 1e-4
        """

        for layer in self.base_model.layers[-trainable_layers:]:
            layer.trainable = True

        self.lr = fine_tune_lr
        self.compile_model()
        print(f"Fine-tuning enabled: Top {trainable_layers} layers are now trainable.")


     
    def compile_model(self):
        """
            If you want to fine-tune the model, please specify the LR, otherwise it is a default form tf.keras.optimisers.Adam(0.001)
        """
        
        
        self.model.compile(loss="sparse_categorical_crossentropy",
                               optimizer=tf.keras.optimizers.Adam(self.lr),
                               metrics=["accuracy"])
        
            


    def save_model(self, path="models/efficientnetb0.keras"):
        """
        Saves the trained model.
        """
        self.model.save(path)
        print(f"Model saved at {path}")


    def load_model(self, path="models/efficientnetb0.keras"):
        """
        Loads a saved model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")


        



