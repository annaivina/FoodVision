import os
from torchvision import datasets
from torch.utils.data import DataLoader 
from typing import Callable, Optional 

"""
Creats the Dataloaders for image classification data. 
Otputs train, test, class names
"""

NUM_WORKERS  = os.cpu_count()

def create_loaders(config,
                   transform: Optional[Callable] = None, #So any transfor can be inserted
                   num_workers: int=NUM_WORKERS
                   ):
    
    """
    Creates training ans testing pytorch datasets.

    Args:
    Config: the config with the train_dir, test_dir, batch_size
    transform: torchvision transforms to perform on training and testing data
    num_workers: integer for number of workers per DataLoader

    Returns:
    A tuple of (train_dataloader, test_dataloader, class_names)
    """

    train_data = datasets.ImageFolder(config['datasets']['train_dir'], transform=transform)
    test_data = datasets.ImageFolder(config['datasets']['test_dir'], transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size = config['datasets']['batch_size'],
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size = config['datasets']['batch_size'],
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names


