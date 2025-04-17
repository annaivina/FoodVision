import os 
import torch
from src.datasets.dataset import create_loaders
from src.datasets.transforms import TorchTransform, AlbumenationTransforms
from src.models.custom_vgg import CustomVgg
from src import engine, import_config, save_model
import argparse

#Setup the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(config_path):

    config = import_config.load_config(config_path)
    
    #Setup the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #Call transformations
    transforms = TorchTransform(config)
    
    #Create Dataloaders:
    train_ds, test_ds, class_names = create_loaders(config, transform=transforms)

    #Create the model 
    model = CustomVgg(config['model']['input_shape'], config['model']['hidden_units'], output_shape=len(class_names)).to(device)

    #Define Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])

    #Train the model 
    engine.train(model, train_ds, test_ds, optimizer, loss_fn, config['datasets']['epochs'], device)

    #Save the model
    save_model.save_model(model, target_dir='models', model_name='custom_vgg_test_1.pth')


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='configs/vgg_config.yaml')

    args = parser.parse_args()
    main(config_path = args.config_path)