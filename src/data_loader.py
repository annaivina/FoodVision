from src.utils import preprocess_image
import tensorflow_datasets as tfds
import tensorflow as tf


class DataLoader:
    def __init__(self, dataset_name='food101', batch_size=32):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_classes = None
        self.class_names = None

    def load_data(self):
        print("Start loading the data set to the folder")
        (train_data, test_data), data_info = tfds.load(name=self.dataset_name,
                                                       split=['train[:10%]', 'validation[:10%]'],
                                                       as_supervised=True,
                                                       shuffle_files=True,
                                                       with_info=True)
        
        self.class_names = data_info.features['label'].names
        self.num_classes = len(self.class_names)

        #Preprocess the data so the image will be resied and will be float32
       
        train_data = train_data.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_data = test_data.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_data, test_data


    
