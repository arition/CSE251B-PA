import pandas
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class XRayDataset(Dataset):
    def __init__(self, csv_path, augmentation=False) -> None:
        super().__init__()
        csv = pandas.read_csv(csv_path)
        self.images = list(csv['StudyInstanceUID'])

        self.ett = torch.zeros((len(self.images), 3), dtype=torch.float)
        self.ngt = torch.zeros((len(self.images), 4), dtype=torch.float)
        self.cvc = torch.zeros((len(self.images), 3), dtype=torch.float)
        self.sgc = torch.zeros((len(self.images), 1), dtype=torch.float)

        self.ett[csv['ETT - Abnormal'] == 1, 0] = 1
        self.ett[csv['ETT - Borderline'] == 1, 1] = 1
        self.ett[csv['ETT - Normal'] == 1, 2] = 1

        self.ngt[csv['NGT - Abnormal'] == 1, 0] = 1
        self.ngt[csv['NGT - Borderline'] == 1, 1] = 1
        self.ngt[csv['NGT - Incompletely Imaged'] == 1, 2] = 1
        self.ngt[csv['NGT - Normal'] == 1, 3] = 1

        self.cvc[csv['CVC - Abnormal'] == 1, 0] = 1
        self.cvc[csv['CVC - Borderline'] == 1, 1] = 1
        self.cvc[csv['CVC - Normal'] == 1, 2] = 1

        self.sgc[csv['Swan Ganz Catheter Present'] == 1, 0] = 1

        self.augmentation = augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(f'data/train/{self.images[index]}.jpg').convert('RGB')
        augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((-90, 90)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if self.augmentation:
            image = augmentation_transform(image)
        else:
            image = transform(image)

        return image, self.ett[index], self.ngt[index], self.cvc[index], self.sgc[index]
