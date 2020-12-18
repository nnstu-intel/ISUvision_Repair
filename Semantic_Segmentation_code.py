import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from Segmentation.mit_semseg.models import *
from Segmentation.mit_semseg.utils import colorEncode

class Segmentator(object):
    def __init__(self):
        self.colors = scipy.io.loadmat('Segmentation/data/color150.mat')['colors']
        self.names = {}
        with open('Segmentation/data/object150_info.csv') as f:
            self.reader = csv.reader(f)
            next(self.reader)
            for row in self.reader:
                self.names[int(row[0])] = row[5].split(";")[0]
        # Network Builders
        self.net_encoder = ModelBuilder.build_encoder(
            arch='resnet50dilated',
            fc_dim=2048,
            weights='Segmentation/encoder_epoch_20.pth')
        self.net_decoder = ModelBuilder.build_decoder(
            arch='ppm_deepsup',
            fc_dim=2048,
            num_class=150,
            weights='Segmentation/decoder_epoch_20.pth',
            use_softmax=True)

        self.crit = torch.nn.NLLLoss(ignore_index=-1)
        self.segmentation_module = SegmentationModule(self.net_encoder, self.net_decoder, self.crit)
        self.segmentation_module.eval()
        self.segmentation_module.cuda()


    def visualize_result(self, img, pred, index=None):
        # filter prediction class if requested
        if index is not None:
            pred = pred.copy()
            pred[pred != index] = -1
            print(f'{self.names[index + 1]}:')

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(numpy.uint8)

        return PIL.Image.fromarray(pred_color)

    def main_get_results(self, input_image):
        pil_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])  # across a large photo dataset.
        ])

        input_image = input_image.convert('RGB')
        img_original = numpy.array(input_image)
        img_data = pil_to_tensor(input_image)
        singleton_batch = {'img_data': img_data[None].cuda()}
        output_size = img_data.shape[1:]

        with torch.no_grad():
            scores = self.segmentation_module(singleton_batch, segSize=output_size)

        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()
        return self.visualize_result(img_original, pred)