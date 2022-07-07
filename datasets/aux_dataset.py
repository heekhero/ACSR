from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class FSDataset(Dataset):
    def __init__(self, num_class, samples_per_class, image_size, return_item=False):

        self.image_path = None
        self.num_class = num_class
        self.samplers_per_class = samples_per_class

        self.image_size = image_size

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trans = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize])

        self.return_item = return_item


    def __len__(self):
        return self.num_class * self.samplers_per_class

    def __getitem__(self, item):
        raw_image = Image.open(self.image_path[item]).convert('RGB')
        augmented_view = self.trans(raw_image)

        label = int(float(item) / self.samplers_per_class)

        if self.return_item:
            return augmented_view, label, item
        else:
            return augmented_view, label