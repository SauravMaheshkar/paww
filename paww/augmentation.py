import albumentations as A

__all__ = ["get_transforms"]


def get_transforms(cfg: dict, data):

    if data == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(cfg.size, cfg.size, scale=(0.85, 1.0)),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(cfg.size, cfg.size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.ToTensorV2(),
            ]
        )
