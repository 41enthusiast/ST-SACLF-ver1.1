from typing import Optional, Callable, Any, Tuple
import glob
import os

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def get_xy(filename, root, sep):
  filenames = []
  fileclasses = []
  for line in open(filename):
    filenames.append(os.path.join(root, line.split(sep)[0]))
    fileclasses.append(line.split(sep)[1].split('\n')[0])
  return filenames, fileclasses


class PACS_Dataset(ImageFolder):
    def __init__(self, filename: str, root: str, sep: str, domain_name, transform: Optional[Callable] = None):
        super().__init__(root, transform)
        self.domain_classes, self.domain_class_to_idx = self.find_classes(self.root)
        self.classes, self.class_to_idx = self.find_classes(self.root+f'/{domain_name}')
        self.filenames, self.fileclasses = get_xy(filename, root, sep)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.loader(self.filenames[index])
        target = int(self.fileclasses[index])
        if self.transform is not None:
            return self.transform(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.filenames)

def get_domain_dl(domain_name: str, transform = transforms.ToTensor(), batch_size: int = 4, data_type: str = 'train'):
    domain_names = ['art_painting', 'cartoon', 'photo', 'sketch']
    domain_labels_path = {domain_name: glob.glob(f'data/{domain_name}_{data_type}_*')[0] for domain_name in domain_names}[domain_name]
    #print(domain_labels_path)
    training_dataset = PACS_Dataset(domain_labels_path,
                                    'data/pacs_data',
                                    ' ',
                                    domain_name,
                                    transform)
    #print(training_dataset.class_to_idx)
    training_dl = DataLoader(training_dataset, batch_size, True)

    return training_dl, training_dataset

if __name__ == '__main__':
    train_samples = []
    domain_names = ['art_painting', 'cartoon', 'photo', 'sketch']

    for i in domain_names:
        training_dl, _ = get_domain_dl(i)
        x, y = next(iter(training_dl))
        print(x.shape, y.shape)
        print(y-1)

    #class wise samples - subset
    _, train_ds = get_domain_dl('cartoon')
    indices = list(filter(lambda idx: train_ds.fileclasses[idx] == '1', range(len(train_ds.fileclasses))))
    dog_pacs = Subset(train_ds, indices)
    print(dog_pacs[0])


