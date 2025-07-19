from torchvision import transforms as T  # noqa: WPS347, WPS111

from src.configs.project_config import DatasetConfig


def get_train_transforms(config: DatasetConfig) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize(config.img_size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=45),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        T.ToTensor(),
        T.Normalize(
            mean=config.mean,
            std=config.std
        )
    ])


def get_valid_transforms(config: DatasetConfig) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize(config.img_size),
        T.ToTensor(),
        T.Normalize(
            mean=config.mean,
            std=config.std
        )
    ])
