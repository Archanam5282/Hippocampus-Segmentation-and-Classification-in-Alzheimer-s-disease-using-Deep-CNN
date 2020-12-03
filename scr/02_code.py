
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from PIL import Image

# Tell Matplotlib to not try and use interactive backend
mpl.use("agg")

def mpl_image_grid(images):
    # Create a figure to contain the plot.
    n = min(images.shape[0], 16) # no more than 16 thumbnails
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2*rows, 2*cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        # Start next subplot.
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            # this is specifically for 3 softmax'd classes with 0 being bg
            # We are building a probability map from our three classes using 
            # fractional probabilities contained in the mask
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0]                             for y in range(vol.shape[2])]                             for x in range(vol.shape[1])]
            plt.imshow(img)
        else: # plotting only 1st channel
            plt.imshow((images[i, 0]*255).int(), cmap= "gray")

    return figure

def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter):

    writer.add_scalar("Loss",                    loss, counter)
    writer.add_figure("Image Data",        mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask",        mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure("Probability map",        mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction",        mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)), global_step=counter)

def save_numpy_as_image(arr, path):

    plt.imshow(arr, cmap="gray") #Needs to be in row,col order
    plt.savefig(path)

def med_reshape(image, new_shape):
D array of desired shape, padded with zeroes


    reshaped_image = np.zeros(new_shape)


    reshaped_image[:image.shape[0],:image.shape[1],:image.shape[2]]=image

    return reshaped_image


import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

#from utils.utils import med_reshape

def LoadHippocampusData(root_dir, y_shape, z_shape):


    image_dir = os.path.join('images')
    label_dir = os.path.join('labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    for f in images:


        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)


        out.append({"image": image, "seg": label, "filename": f})

    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)



import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):

    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):

        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        img=self.data[slc[0]]['image'][slc[1]]
        seg=self.data[slc[0]]['seg'][slc[1]][None,:]
        sample['image']=torch.from_numpy(img).unsqueeze(0).cuda()
        sample['seg']=torch.from_numpy(seg).long().cuda()

        return sample

    def __len__(self):

        return len(self.slices)

import numpy as np

def Dice3d(a, b):

    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")


    
    a[a>0]=1
    b[b>0]=1
    intersection = np.sum(a*b)
    volumes = np.sum(a) + np.sum(b)
    if volumes == 0:
        return -1

    return 2.*float(intersection) / float(volumes)

def Jaccard3d(a, b):

    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")


    a[a>0]=1
    b[b>0]=1
    intersection = np.sum(a*b)
    union = np.sum(a) + np.sum(b)-intersection

    if union == 0:
        return -1
    return float(intersection) / float(union)


def sensitivity(gt,pred):
    # Sens = TP/(TP+FN)
    tp = np.sum(gt[gt==pred])
    fn = np.sum(gt[gt!=pred])

    if fn+tp == 0:
        return -1

    return (tp)/(fn+tp)



import torch

from torch import nn

class UNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4, norm_layer=nn.InstanceNorm2d):
        # norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNet, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-1), out_channels=initial_filter_size * 2 ** num_downs,
                                             num_classes=num_classes, kernel_size=kernel_size, norm_layer=norm_layer, innermost=True)
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(in_channels=initial_filter_size * 2 ** (num_downs-(i+1)),
                                                 out_channels=initial_filter_size * 2 ** (num_downs-i),
                                                 num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(in_channels=in_channels, out_channels=initial_filter_size,
                                             num_classes=num_classes, kernel_size=kernel_size, submodule=unet_block, norm_layer=norm_layer,
                                             outermost=True)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        # downconv
        pool = nn.MaxPool2d(2, stride=2)
        conv1 = self.contract(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)
        conv2 = self.contract(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, norm_layer=norm_layer)

        # upconv
        conv3 = self.expand(in_channels=out_channels*2, out_channels=out_channels, kernel_size=kernel_size)
        conv4 = self.expand(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

        if outermost:
            final = nn.Conv2d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels,
                                        kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose2d(in_channels*2, in_channels, kernel_size=2, stride=2)

            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm2d):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(layer, target_width, target_height):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            crop = self.center_crop(self.model(x), x.size()[2], x.size()[3])
            return torch.cat([x, crop], 1)


import torch
import numpy as np



class UNetInferenceAgent:
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):

        patch_size = 64
        volume=(volume-volume.min())/(volume.max()-volume.min())
        volume = med_reshape(volume, new_shape=(volume.shape[0], patch_size, patch_size))
        
        masks=np.zeros(volume.shape)
        for slice_idx in range(masks.shape[0]):
            # normalize the image
            slice_0 = volume[slice_idx,:,:]
            #slice0_norm = (slice0-slice0.min())/(slice0.max()-slice0.min())
            data=torch.from_numpy(slice_0).unsqueeze(0).unsqueeze(0).float().to(self.device)
            pred=self.model(data)
            pred=np.squeeze(pred.cpu().detach())
            pred=pred.argmax(axis=0)
            masks[slice_idx,:,:]=pred
        return masks

    def single_volume_inference(self, volume):

        self.model.eval()


        slices = []


        
        masks=np.zeros(volume.shape)
        for slice_idx in range(masks.shape[0]):
            slice_0 = volume[slice_idx,:,:]
            data=torch.from_numpy(slice_0).unsqueeze(0).unsqueeze(0).float().to(self.device)
            pred=self.model(data)
            pred=np.squeeze(pred.cpu().detach())
            pred=pred.argmax(axis=0)
            masks[slice_idx,:,:]=pred
        return masks


import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from data_prep.SlicesDataset import SlicesDataset
#from utils.utils import log_to_tensorboard
#from utils.volume_stats import Dice3d, Jaccard3d, sensitivity
#from networks.RecursiveUNet import UNet
#from inference.UNetInferenceAgent import UNetInferenceAgent

class UNetExperiment:

    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)

 
        self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)

        self.test_data = dataset[split["test"]]

        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.model = UNet(num_classes=3)
        self.model.to(self.device)


        self.loss_function = torch.nn.CrossEntropyLoss()


        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

      

        
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def train(self):
        """
        This method is executed once per epoch and takes 
        care of model weight update cycle
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        # Loop over our minibatches
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()


            data = batch['image'].to(self.device, dtype=torch.float)
            target = batch['seg'].to(self.device)

            prediction = self.model(data)

            prediction_softmax = F.softmax(prediction, dim=1)

            loss = self.loss_function(prediction, target[:, 0, :, :])


            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                # Output to console on every 10th batch
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete")

                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter)

            print(".", end='')

        print("\nTraining complete")

    def validate(self):

        print(f"Validating epoch {self.epoch}...")

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                
                # TASK: Write validation code that will compute loss on a validation sample
                # <YOUR CODE HERE>
                data = batch['image'].to(self.device, dtype=torch.float)
                target = batch['seg'].to(self.device)
                prediction = self.model(data)

                prediction_softmax = F.softmax(prediction, dim=1)

                loss = self.loss_function(prediction, target[:, 0, :, :])
                


                print(f"Batch {i}. Data shape {data.shape} Loss {loss}")

                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax, 
            prediction,
            (self.epoch+1) * 100)
        print(f"Validation complete")

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")

        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        print("Testing...")
        self.model.eval()

        
        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []
        sen_list =[]

        # for every in test set
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])



            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            sen = sensitivity(x["seg"],pred_label)
            dc_list.append(dc)
            jc_list.append(jc)
            sen_list.append(sen)


            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc,
                "sensitivity": sen
                })
            print(f"{x['filename']} Dice {dc:.4f}. {100*(i+1)/len(self.test_data):.2f}% complete")

        out_dict["overall"] = {
            "mean_dice": np.mean(dc_list),
            "mean_jaccard": np.mean(jc_list),
            "mean_sensitivity": np.mean(sen_list)
            }

        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        print("Experiment started.")

        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()

        # save model for inferencing
        self.save_model_parameters()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")





"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import random
from sklearn.model_selection import train_test_split



class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"out/"
        self.n_epochs = 2
        self.learning_rate = .0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "result/"

if __name__ == "__main__":

    c = Config()


    print("Loading data...")

    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)



    keys = range(len(data))


    split = dict()

    
    split['train'],split['test'] = train_test_split(keys, test_size =0.25, random_state=40)
    split['train'],split['val'] = train_test_split(split['train'], test_size =0.25, random_state=40)
    print('Split Done')
    

    exp = UNetExperiment(c, split, data)

    
    exp.run()

    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
