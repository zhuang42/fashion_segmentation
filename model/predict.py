import torch
import argparse
import os
from dataloaders.utils import decode_seg_map_sequence
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import time



#Deeplabv3+
from modeling.deeplab import DeepLab
#Deeplabv3
from modeling.my_deeplab import My_DeepLab
#Unet
from modeling.unet import UNet

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


def transform_val(sample):
    composed_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    return composed_transforms(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSC420 Segmentation")

    parser.add_argument('--model', '-m', default='deeplabv3+',
                        metavar='[deeplabv3+, deeplabv3, unet]',
                        help="Specify Which Model"
                             "(default : DeepLabV3+)",
                        choices = ['deeplabv3+', 'deeplabv3', 'unet']
                        )


    parser.add_argument('--task', '-t', metavar='[person, fashion]',
                            help = "Specify Task [person, fashion]",
                            choices= ['person', 'clothes'], required=True
                        )


    parser.add_argument('--path', '-p',
                        metavar='model_path', help="Specify Model Path")

    parser.add_argument('--input', '-i', metavar='input_path',
                        help='Input image ', required=True)

    parser.add_argument('--output', '-o', metavar='output_path',
                        help='Output image', required=True)



    args = parser.parse_args()

    path = args.path


    if args.task == 'person':
        dataset = "fashion_person"
        path = model_paths['person'][args.model]
        nclass = 2
    elif args.task == 'clothes':
        dataset = "fashion_clothes"
        path = model_paths['clothes'][args.model]
        nclass = 7


    if (args.path):
        path = args.path

    print("Model Path is {}".format(path))


    if args.model == "deeplabv3+":
        #Suggested in paper, output stide is set to 8
        #to get better evaluation performance
        model = DeepLab(num_classes=nclass, output_stride=8)
    elif args.model == 'deeplabv3':

        model = My_DeepLab(num_classes=nclass, in_channels=3)
    elif args.model == 'unet':
        model = UNet(num_filters = 32, num_categories=nclass, num_in_channels=3)



    if torch.cuda.is_available():
        print("Moving model to GPU")
        model.cuda()
    else:
        print("CUDA not available, run model on CPU")
        model.cpu()
        torch.set_num_threads(8)

    if not os.path.isfile(path):
        raise RuntimeError("no model found at'{}'".format(path))

    if not os.path.isfile(args.input):
        raise RuntimeError("no image found at'{}'".format(input))

    if os.path.exists(args.output):
        raise RuntimeError("Existed file or dir found at'{}'".format(args.output))


    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])

    print("Model loaded")



    img = Image.open(args.input).convert('RGB').resize((400, 600), Image.BILINEAR)
    img = transform_val(img)


    print("Image Loaded")
    model.eval()
    t = time.time()

    print("Start Processing")

    with torch.no_grad():
        img = torch.unsqueeze(img, 0)
        output = model(img)
        prediction = decode_seg_map_sequence(torch.max(output[:], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset)

        prediction = prediction.squeeze(0)
        save_image(prediction, args.output, normalize=False)

    print("Time spend {}s".format(time.time() - t))


