import torch.utils.data
from .base_data_loader import BaseDataLoader


def CreateDataset(opt, src, ref):
    from .aligned_dataset_rand_seg_onlymap import AlignedDataset_Rand_Seg_onlymap
    dataset = AlignedDataset_Rand_Seg_onlymap()
    dataset.initialize(opt, src, ref)

    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, src, ref):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, src, ref)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True)
        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=opt.batchSize,
        #     shuffle=not opt.serial_batches,
        #     num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

    def __getitem__(self, index):
        for i, data in enumerate(self.dataloader):
            if index == i:
                return data
