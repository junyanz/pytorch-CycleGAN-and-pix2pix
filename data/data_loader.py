
def CreateDataLoader(opt):
    data_loader = None
    if opt.align_data > 0:
        from data.aligned_data_loader import AlignedDataLoader
        data_loader = AlignedDataLoader()
    else:
        from data.unaligned_data_loader import UnalignedDataLoader
        data_loader = UnalignedDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
