from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


from config import DATA_PATH

class MiniImagenet(Dataset):
    def __init__(self, subset, state, local_rank=0, world_size=1, shuffle_indices=None, **kwargs):
        self.kwargs = kwargs

        self.subset = subset
        self.state = state

        #spilt the dataset for each ensemble model according to the local_rank
        meta_array = self.index_subset(DATA_PATH, self.subset)
        if shuffle_indices is not None:
            meta_array = list(np.array(meta_array)[shuffle_indices.numpy()])

        len_of_meta_array = len(meta_array)
        self.index_begin = int(float(local_rank)/float(world_size) * len_of_meta_array)
        if local_rank != world_size-1:
            local_meta_array = meta_array[int(float(local_rank)/float(world_size) * len_of_meta_array) : int(float(local_rank+1)/float(world_size) * len_of_meta_array)]
        else:
            local_meta_array = meta_array[int(float(local_rank) / float(world_size) * len_of_meta_array):]

        self.df = pd.DataFrame(local_meta_array)
        print('local_rank : {}    df_size : {}'.format(local_rank, len(self.df)))
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.id_to_class_name = {i : self.unique_characters[i] for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if 'center' in state:
            trans = transforms.Compose([
                transforms.Resize(size=int(kwargs['image_size'] * 1.15)),
                transforms.CenterCrop(kwargs['image_size']),
                transforms.ToTensor(),
                normalize])
        elif 'random_crop' in state:
            trans = transforms.Compose([
                transforms.RandomResizedCrop(size=int(kwargs['image_size'])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            raise NotImplementedError

        self.trans = trans

    def __getitem__(self, item):
        raw_image = Image.open(self.datasetid_to_filepath[item])
        augmented_view = self.trans(raw_image)

        return_list = [augmented_view]
        if 'id' in self.state:
            return_list.append(item)

        if 'target' in self.state:
            return_list.append(self.datasetid_to_class_id[item])

        return return_list

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(root_path, subset):
        images = []
        print('Indexing {}...'.format(subset))
        subset_len = 0
        for root, folders, files in os.walk(root_path + '/miniImagenet/{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(root_path + '/miniImagenet/{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        images = sorted(images, key=lambda x:x['filepath'])
        return images