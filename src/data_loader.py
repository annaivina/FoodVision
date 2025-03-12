#from src.utils import preprocess_image - Old preprocessing image file. This has been substituted to Albumentations
import albumentations as A
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
       
        train_data = train_data.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_data = test_data.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_data, test_data
    

    #Implementing new libriary for data augmentation 
    @staticmethod
    def apply_transform(image):

        img_size = 224
        transformations = A.Compose(
            [
                #Resise only the EfficientNet does its own rescaling to 1/255
                A.Resize(img_size, img_size),
                A.OneOf([A.HorizontalFlip(),
                         A.VerticalFlip()], p=0.3), #percent of times it iwll happen 
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=30, p=0.3),
                A.Sharpen(alpha=(0.2,0.3), lightness=(0.5,1), p=0.5)
            ]
        )

        #image = image.numpy()
        image = transformations(image=image)['image']
        
        return tf.cast(image, tf.float32)
    
    @staticmethod
    def preprocess_image(image, label):
        image = tf.numpy_function(func=DataLoader.apply_transform, inp=[image], Tout=tf.float32)
        image.set_shape((224,224,3))
        return image, label
        
    
 
    
