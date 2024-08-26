import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
import PIL
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")
global kernel_sizes 

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, dataset='EvCSLRDataset', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="video", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224, use_eventImage=True):
        self.mode = mode  # train or not-train
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
       
        self.feat_prefix = f""
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()

        self.resize = 256
        self.use_eventImage = use_eventImage
      

    def __getitem__(self, idx):
        input_data, label, fi = self.read_video(idx)
        input_data = [np.array(Image.fromarray(d).resize((256, 256))) for d in input_data]
        RD = random.random()
        
        crop_h, crop_w = 224,224
        
        if isinstance(input_data[0], np.ndarray):
            im_h, im_w, im_c = input_data[0].shape
        elif isinstance(input_data[0], PIL.Image.Image):
            im_w, im_h = input_data[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(input_data[0])))

        w1 = random.randint(0, im_w - crop_w)
        h1 = random.randint(0, im_h - crop_h)

        if isinstance(input_data, list):
            input_data_t = np.array(input_data)
            tmp = torch.from_numpy(input_data_t.transpose((0, 3, 1, 2))).float()
        if isinstance(input_data, np.ndarray):
            tmp = torch.from_numpy(input_data_t.transpose((0, 3, 1, 2)))
        min_len = 32
        max_len = int(np.ceil(230/1))
        L = 1.0 - 0.2
        U = 1.0 + 0.2
        vid_len = len(tmp)
       
        new_len = int(vid_len * (L + (U - L) * np.random.random()))
       
        if new_len < min_len:
            new_len = min_len
        if new_len > max_len:
            new_len = max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))

        input_data, label = self.data_aug(input_data, label, None,RD,w1,h1,index)
    

        if self.use_eventImage:
            e1, e3 = self.read_event_repre(idx)
            e1 = [np.array(Image.fromarray(d).resize((256, 256))) for d in e1]
            e3 = [np.array(Image.fromarray(d).resize((256, 256))) for d in e3]
        
        e1, label = self.data_aug(e1, label, None,RD,w1,h1,index)    
        e3, label = self.data_aug(e3, label, None,RD,w1,h1,index)
      
        return (input_data / 127.5 - 1, torch.LongTensor(label),
                self.inputs_list[idx]['original_info'], e1 / 127.5 - 1,
                e3 / 127.5 - 1)
        

    def read_video(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
          
            img_folder = os.path.join(fi['folder']) # ori_RGB

            img_folder = img_folder[0:-6] + f'_low/*.png'  # low_light
     
        elif self.dataset == 'CSL':
            img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.jpg")
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, fi['folder'])
        elif self.dataset == 'EvCSLRDataset':
            img_folder = os.path.join(fi['folder'])

        img_list = sorted(glob.glob(img_folder))
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list = []
        
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi
    
    def read_event_repre(self, index):
        # load file info
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
          
            img_folder = os.path.join(fi['folder']) # ori_RGB
        elif self.dataset == 'EvCSLRDataset' or self.dataset == 'ourdataset':
            img_folder = os.path.join(fi['folder'])
        
        e1_path = img_folder[0:-13]+f'/eventVoxel_5'
        e3_path = img_folder[0:-13]+f'/eventFrame_eventAdd'

        e1_list = []
        e3_list = []

        event_list = sorted(os.listdir(e1_path))
        for event_name in event_list:
            if event_name != ".ipynb_checkpoints":
                e1_folder_path = e1_path + '/' + event_name
                e3_folder_path = e3_path + '/' + event_name
                e1_list.append(e1_folder_path)
                e3_list.append(e3_folder_path)
        e1_list.append(e1_list[-1])
        e3_list.append(e3_list[-1])
        
        
        return ([cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in e1_list],
                [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in e3_list])

    def read_event(self, index):
        # read event data
        fi = self.inputs_list[index]
        if 'phoenix' in self.dataset:
            event_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['event_folder'])
        elif self.dataset == 'EvCSLRDataset':
            
            event_folder = os.path.join(fi['original_info'].split('|')[-1])
            event_path = event_folder[0:-6]

        event_data = np.load(event_path+f'/event.npz')
        e1 = event_data['arr1']
        
        e1.resize((e1.shape[0], 5, 256, 256))
       
        return torch.tensor(e1)

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    

    @staticmethod
    def collate_fn(batch):
       
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info ,e1, e3= list(zip(*batch))
        
        left_pad = 0
        last_stride = 1
        total_stride = 1
        global kernel_sizes 
        kernel_sizes = ['K5', "P2", 'K5', "P2"]
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        if len(video[0].shape) > 3:
          
            max_len = len(video[0])
           
            video_length = torch.LongTensor([np.ceil(len(vid) / total_stride) * total_stride + 2*left_pad for vid in video])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)

            padded_event1 = [torch.cat(
                (
                    e[0][None].expand(left_pad, -1, -1, -1),
                    e,
                    e[-1][None].expand(max_len - len(e) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for e in e1]
            padded_event1 = torch.stack(padded_event1)
           
            padded_event3 = [torch.cat(
                (
                    e[0][None].expand(left_pad, -1, -1,-1),
                    e,
                    e[-1][None].expand(max_len - len(e) - left_pad, -1, -1,-1),
                )
                , dim=0)
                for e in e3]
            padded_event3 = torch.stack(padded_event3)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info, padded_event1, padded_event3

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

