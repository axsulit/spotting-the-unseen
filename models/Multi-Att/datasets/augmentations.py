from albumentations import *
# No need to import from imgaug now, using albumentations' GaussianNoise instead.
# from imgaug.augmenters import IAAAdditiveGaussianNoise  <-- REMOVE THIS

# Define augmentations
augment0 = Compose([HorizontalFlip()], p=1)
"""Basic augmentation with only horizontal flip."""

augment1 = Compose([HorizontalFlip(), HueSaturationValue(p=0.5), RandomBrightnessContrast(p=0.5)], p=1)
"""Medium augmentation with horizontal flip, color adjustments, and brightness/contrast changes."""

augment_rand1 = Compose([RandomCrop(380, 380), HorizontalFlip(), HueSaturationValue(p=0.5), RandomBrightnessContrast(p=0.5)], p=1)
"""Random crop augmentation with color and brightness adjustments."""

augment2 = Compose([HorizontalFlip(), HueSaturationValue(p=0.5), RandomBrightnessContrast(p=0.5),
                    OneOf([
                        GaussNoise(p=0.3),  # Replaced IAAAdditiveGaussianNoise with GaussianNoise from albumentations
                        GaussNoise(),
                    ], p=0.3),
                    OneOf([
                        MotionBlur(),
                        GaussianBlur(),
                    ], p=0.3), ToGray(p=0.1)], p=1)
"""Strong augmentation combining multiple transformations including noise, blur, and color changes."""

# Create a dictionary of augmentations
augmentations = {'augment0': augment0, 'augment1': augment1, 'augment2': augment2, 'augment_rand1': augment_rand1}
"""Dictionary mapping augmentation names to their respective transformation pipelines."""
