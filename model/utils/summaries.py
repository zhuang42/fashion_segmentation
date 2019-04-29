import os
import torch
import numpy as np
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

from mypath import  Path


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        print(os.path.join(self.directory))
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)

    def visualize_pregt(self, writer, dataset, image, target, output, val_batch_i):
        assert image.size(0) == target.size(0) == output.size(0)
        batch_size = image.size(0)

        predicteds = decode_seg_map_sequence(torch.max(output[:], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset)

        gts = decode_seg_map_sequence(torch.squeeze(target[:], 1).detach().cpu().numpy(),
                                                       dataset=dataset)

        #Have same size
        assert predicteds.size() == gts.size()

        for i in range(batch_size):
            predicted = predicteds[i]
            gt = gts[i]
            combine = np.concatenate((gt, predicted), axis=2)
            writer.add_image('Results', combine, val_batch_i * batch_size + i + 1)


    def save_pred(self, dataset, output, val_batch_i):
        output_dir = os.path.join(os.path.join(self.directory), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        batch_size = output.size(0)
        predicteds = decode_seg_map_sequence(torch.max(output[:], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset)

        for i in range(batch_size):
            #validation set start from 301
            idx = 300 + val_batch_i * batch_size + i + 1
            file_save_path = os.path.join(output_dir, str(idx).zfill(4) + '.png')
            predicted = predicteds[i]
            save_image(predicted, file_save_path, normalize=False)


