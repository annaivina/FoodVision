# FoodVision
Here is the CNN algo-based  EffB0 net for classifying the Food Pictures.\
Version allows to run the original model --> fine tune with the configurable parameters in configs.yaml and do predictions. 

It can also be optimized to use mixed_precision training. 

To run the code, use run.sh script (not yet possible to run on GPU machines)

Additionally, I have created the pytorch impelemntation of the training using the custom CNN model. The code is available in pytorch_impl.
To download the dataset one can use pytorch_impl/download_data.py and use this data to train any model in PyToarch or in TensorFlow. 
