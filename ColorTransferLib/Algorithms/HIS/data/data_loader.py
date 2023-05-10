
def CreateDataLoader(opt, src, ref):
    from ColorTransferLib.Algorithms.HIS.data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, src, ref)
    return data_loader
