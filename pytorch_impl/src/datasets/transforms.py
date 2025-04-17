import numpy as np
from torchvision import transforms 
import albumentations as alb
from albumentations.pytorch import ToTensorV2


class AlbumenationTransforms:
    def __init__(self, config):
        img_size = config['datasets']['img_size']
        self.transform  = alb.Compose([
            alb.Resize(img_size, img_size),
            alb.HorizontalFlip(p=0.5),
            ToTensorV2(),#scales to [0,1] and does (C,H,W)
            ])
        
    def __call__(self, image):
        image = np.array(image)#albumentations work with np arrays, cant use raw imagfes 
        return self.transform(image=image)['image']
    
class TorchTransform:
    def __init__(self, config):
        img_size = config['datasets']['img_size']
        self.transform = transforms.Compose([ 
                                    transforms.Resize((img_size, img_size)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor()])
        
    def __call__(self, image):
        return self.transform(image)

        