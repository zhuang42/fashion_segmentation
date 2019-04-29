import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from modeling.deeplab import *
from modeling.my_deeplab import  My_DeepLab
from modeling.unet import UNet


from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

from datasets import fashion

from torch.utils.data import Dataset, DataLoader
from torch import nn

model_paths = {
    'person': {
        'deeplabv3+': "./bestmodels/deep_person/checkpoint.pth.tar",
        'deeplabv3': "./bestmodels/my_deep_person/checkpoint.pth.tar",
        'unet': "./bestmodels/unet_person/checkpoint.pth.tar"

    },
    'clothes': {
        'deeplabv3+': "./bestmodels/deep_clothes/checkpoint.pth.tar",
        'deeplabv3': "./bestmodels/my_deep_clothes/checkpoint.pth.tar",
        'unet': "./bestmodels/unet_clothes/checkpoint.pth.tar"
    }
}




def set_parameter_requires_grad(model, flag):
    for param in model.parameters():
        param.requires_grad = flag


class MyTrainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}


        if (args.dataset == "fashion_person"):

            train_set = fashion.FashionDataset(args, Path.db_root_dir("fashion_person"), mode='train',type = 'person')
            val_set = fashion.FashionDataset(args, Path.db_root_dir("fashion_person"), mode='test', type='person')
            self.nclass = train_set.nclass



            print("Train size {}, val size {}".format(len(train_set), len(val_set)))


            self.train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                   **kwargs)
            self.val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False,
                                   **kwargs)
            self.test_loader = None

            assert self.nclass == 2

        elif (args.dataset == "fashion_clothes"):
            train_set = fashion.FashionDataset(args, Path.db_root_dir("fashion_clothes"), mode='train', type='clothes')
            val_set = fashion.FashionDataset(args, Path.db_root_dir("fashion_clothes"), mode='test', type='clothes')
            self.nclass = train_set.nclass

            print("Train size {}, val size {}".format(len(train_set), len(val_set)))

            self.train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                                           **kwargs)
            self.val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False,
                                         **kwargs)
            self.test_loader = None

            assert self.nclass == 7



        #self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        # model = DeepLab(num_classes=self.nclass,
        #                 backbone=args.backbone,
        #                 output_stride=args.out_stride,
        #                 sync_bn=args.sync_bn,
        #                 freeze_bn=args.freeze_bn)
        # Using original network to load pretrained and do fine tuning


        self.best_pred = 0.0

        if args.model == 'deeplabv3+':
            model = DeepLab(backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

            # Loading pretrained VOC model
            if args.resume is not None:
                if not os.path.isfile(args.resume):
                    raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
                if args.cuda:
                    checkpoint = torch.load(args.resume)
                else:
                    checkpoint = torch.load(args.resume,map_location='cpu')
                args.start_epoch = checkpoint['epoch']

                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))

            #Freez the backbone
            if args.freeze_backbone:
                set_parameter_requires_grad(model.backbone, False)

            ######NEW DECODER######
            #Different type of FT
            if args.ft_type == 'decoder':
                set_parameter_requires_grad(model, False)
                model.decoder = build_decoder(self.nclass, 'resnet', nn.BatchNorm2d)
                train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                                {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

            elif args.ft_type == 'last_layer':
                set_parameter_requires_grad(model, False)
                model.decoder.last_conv[8] = nn.Conv2d(in_channels=256, out_channels=self.nclass, kernel_size=1)
                model.decoder.last_conv[8].reset_parameters()
                train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                                {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
            if args.ft_type == 'all':
                #Reset last layer, to generate output we want
                model.decoder.last_conv[8] = nn.Conv2d(in_channels=256, out_channels=self.nclass, kernel_size=1)
                model.decoder.last_conv[8].reset_parameters()

                train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]



            # Define Optimizer
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)


        elif args.model == "unet":
            model = UNet(num_categories=self.nclass, num_filters=args.num_filters)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)

        elif args.model == 'mydeeplab':

            model = My_DeepLab(num_classes=self.nclass, in_channels=3)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)


        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
            print("weight is {}".format(weight))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        self.model, self.optimizer = model, optimizer




        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            # TODO, ADD PARALLEL SUPPORT (NEED SYNC BATCH)
            # self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            # patch_replication_callback(self.model)
            self.model = self.model.cuda()

        args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def visulize_validation(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            #current_index_val_set
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            #we have image, target, output on GPU
            #j, index of image in batch

            self.summary.visualize_pregt(self.writer, self.args.dataset, image, target, output, i)

            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Visualizing:')
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Final Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

    def output_validation(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            #current_index_val_set
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            #we have image, target, output on GPU
            #j, index of image in batch

            #image save
            self.summary.save_pred(self.args.dataset, output, i)

            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Visualizing:')
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Final Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)


    def _load_model(self, path):
        if self.args.cuda:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')

        self.model.load_state_dict(checkpoint['state_dict'])


    def train_loop(self):
        try:
            for epoch in range(self.args.start_epoch, self.args.epochs):
                self.training(epoch)
                if not self.args.no_val and epoch % self.args.eval_interval == (self.args.eval_interval - 1):
                    self.validation(epoch)
        except KeyboardInterrupt:
            print('Early Stopping')
        finally:
            self.visulize_validation()
            self.writer.close()



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == "__main__":
    args = AttrDict()


    ############DeepLabV3+ clothes###########
    args_dict = {
        'model':'deeplabv3+',
        'workers':4,
        'dataset':'fashion_clothes',
        'batch_size':4,
        'resume': './deeplab-resnet.pth.tar',
        'start_epoch': 0,
        'lr':0.007,
        'momentum':0.9,
        'weight_decay':0,
        'nesterov':True,
        'cuda':True,
        'lr_scheduler':'poly',
        'epochs':15,
        'eval_interval': 1,

        #Default Parameter
        'checkname': 'fashion-deeplabv3p-resnet-ft-all',
        'backbone': 'resnet',
        'sync_bn':False,
        'freeze_bn':False,
        'out_stride':16,
        'loss_type':'ce',
        'base_size': 600,
        'crop_size':400,
        'use_balanced_weights':True,
        'no_val':False,
        'ft_type':'all', #last Layer
        'freeze_backbone':False
    }
    #
    args.update(args_dict)

    trainer = MyTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_loop()
    # trainer._load_model(model_paths['clothes']['deeplabv3+'])
    # trainer.output_validation()
    trainer.writer.close()



    ##########DeepLabv3+ PERSON############
    args_dict = {
        'model':'deeplabv3+',
        'workers':4,
        'dataset':'fashion_person',
        'batch_size':4,
        'resume': './deeplab-resnet.pth.tar',
        'start_epoch': 0,
        'lr':0.007,
        'momentum':0.9,
        'weight_decay':5e-4,
        'nesterov':True,
        'cuda':True,
        'lr_scheduler':'poly',
        'epochs':15,
        'eval_interval': 1,

        #Default Parameter
        'checkname': 'fashion-deeplabv3p-resnet-ft-all',
        'backbone': 'resnet',
        'sync_bn':False,
        'freeze_bn':False,
        'out_stride':16,
        'loss_type':'ce',
        'base_size': 600,
        'crop_size':400,
        'use_balanced_weights':True,
        'no_val':False,
        'ft_type':'all', #last Layer
        'freeze_backbone':False
    }
    args.update(args_dict)

    trainer = MyTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_loop()
    # trainer._load_model(model_paths['person']['deeplabv3+'])
    # trainer.output_validation()
    trainer.writer.close()


    ######UNET CLOTHES###########
    args_dict = {
        'model':'unet',
        'workers':4,
        'data_path':'../data/fashion/',
        'dataset':'fashion_clothes',
        'batch_size':4,
        'num_filters':32,
        'start_epoch': 0,
        'lr':0.007,
        'weight_decay':0,
        'cuda': True,
        'lr_scheduler': 'poly',
        'epochs': 100,
        'eval_interval': 5,


        #Default and useless parameters
        #Adam does not have momentum
        'momentum': 0.9,
        'nesterov':True,
        'checkname' : 'fashion-unet',
        'backbone' : 'resnet',
        'sync_bn':False,
        'freeze_bn':False,
        'loss_type':'ce',
        'out_stride': 16,
        'base_size': 600,
        'crop_size':400,
        'use_balanced_weights':True,
        'no_val':False,
        'freeze_backbone':False
    }

    args.update(args_dict)

    trainer = MyTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_loop()
    # trainer._load_model(model_paths['clothes']['unet'])
    # trainer.output_validation()
    trainer.writer.close()

    #######UNET PERSON###########
    args_dict = {
        'model': 'unet',
        'workers': 4,
        'data_path': '../data/fashion/',
        'dataset': 'fashion_person',
        'batch_size': 4,
        'num_filters': 32,
        'start_epoch': 0,
        'lr': 0.007,
        'weight_decay': 0,
        'cuda': True,
        'lr_scheduler': 'poly',
        'epochs': 100,
        'eval_interval': 5,

        # Default and useless parameters
        'momentum': 0.9,
        'nesterov': True,
        'checkname': 'fashion-unet',
        'backbone': 'resnet',
        'sync_bn': False,
        'freeze_bn': False,
        'loss_type': 'ce',
        'out_stride': 16,
        'base_size': 600,
        'crop_size': 400,
        'use_balanced_weights': True,
        'no_val': False,
        #no backbone
        'freeze_backbone': False
    }

    args.update(args_dict)

    trainer = MyTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_loop()
    # trainer._load_model(model_paths['person']['unet'])
    # trainer.output_validation()
    trainer.writer.close()




    #########DEEPLABV3 PERSON############
    args_dict = {
        'model': 'mydeeplab',
        'workers':4,
        'dataset':'fashion_person',
        'batch_size':4,
        'start_epoch': 0,
        'lr':0.007,
        'momentum':0.9,
        'weight_decay':0,
        'nesterov':True,
        'cuda':True,
        'lr_scheduler':'poly',
        'epochs':30,
        'eval_interval': 1,
        'checkname' : 'fashion-deeplabv3-resnet',
        'loss_type':'ce',
        'no_val': False,
        'use_balanced_weights': True,



        # can't finetune my deeplab
        'ft_type': 'all',
        # Can't user resume
        'resume': './deeplab-resnet.pth.tar',
        # Useless argument, since we dont crop the input image
        'base_size': 600,
        'crop_size':400,
        'sync_bn': False,
        'freeze_bn': False,
        'out_stride': 8,
        'freeze_backbone': False,
        'backbone': 'resnet',
    }
    args.update(args_dict)

    trainer = MyTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_loop()

    # trainer._load_model(model_paths['person']['deeplabv3'])
    # trainer.output_validation()
    trainer.writer.close()


    #########DEEPLABV3 CLOTHES############
    args_dict = {
        'model': 'mydeeplab',
        'workers':4,
        'dataset':'fashion_clothes',
        'batch_size':4,
        'start_epoch': 0,
        'lr':0.007,
        'momentum':0.9,
        'weight_decay':0,
        'nesterov':True,
        'cuda':True,
        'lr_scheduler':'poly',
        'epochs':30,
        'eval_interval': 1,
        'checkname' : 'fashion-deeplabv3-resnet',
        'loss_type':'ce',
        'no_val': False,
        'use_balanced_weights': True,



        # can't finetune my deeplab
        'ft_type': 'all',
        # Can't user resume
        'resume': './deeplab-resnet.pth.tar',
        # Useless argument, since we dont crop the input image
        'base_size': 600,
        'crop_size':400,
        'sync_bn': False,
        'freeze_bn': False,
        'out_stride': 8,
        'freeze_backbone': False,
        'backbone': 'resnet',
    }

    args.update(args_dict)

    trainer = MyTrainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    trainer.train_loop()

    # trainer._load_model(model_paths['clothes']['deeplabv3'])
    # trainer.output_validation()
    trainer.writer.close()
