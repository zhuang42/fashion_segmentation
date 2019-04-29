class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/Users/zichunzhuang/Desktop/420Project/data/voc/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/Users/zichunzhuang/Desktop/420Project/data/coco/'

        elif dataset == 'fashion_person':
            return "../data/fashion_person/"
        elif dataset == 'fashion_clothes':
            return "../data/fashion_clothes/"
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
