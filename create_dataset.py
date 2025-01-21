from PIL import Image
import torchvision.transforms.functional as TF
import os
import torch
import numpy as np
from natsort import natsorted
from models.common import RGB2YCrCb
from torch.utils import data
from torchvision import transforms


to_tensor = transforms.Compose([transforms.ToTensor(),
                                ])


class MSRS_Data(data.Dataset):
    def __init__(self, opts, transform=to_tensor):
        super().__init__()
        self.folder = os.path.join(opts.dataroot, 'train')
        dirname = os.listdir(self.folder)
        for sub_dir in dirname:
            temp_path = os.path.join(self.folder, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path
            elif sub_dir == 'vi':
                self.vis_path = temp_path
            elif sub_dir == 'mask':
                self.mask_path = temp_path


        self.name_list = os.listdir(self.inf_path)
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]

        inf_image = Image.open(os.path.join(self.inf_path, name)).convert('L')
        vis_image = Image.open(os.path.join(self.vis_path, name))
        mask_image = Image.open(os.path.join(self.mask_path, name))

        vis_image = np.array(vis_image)
        inf_image = np.array(inf_image)[:,:,np.newaxis]
        mask_image = np.array(mask_image)[:,:,np.newaxis]

        img = np.concatenate([inf_image,vis_image,mask_image],axis=2)
        image = self.transform(img)
        inf_image, vis_image, mask_image = torch.split(image,[1,3,1],dim=0)

        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image,mask_image, name

    def __len__(self):
        return len(self.name_list)



    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            if vis_flage:
                img = Image.open(path).convert('RGB')
                im_ts = TF.to_tensor(img).unsqueeze(0)
            else:
                img = Image.open(path).convert('L') 
                im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts
    

class FusionData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """
    
    def __init__(self, opts, crop=lambda x: x):
        super(FusionData, self).__init__()          
        self.vis_folder = os.path.join(opts.dataroot, opts.dataname, 'test', 'vi')
        self.ir_folder = os.path.join(opts.dataroot, opts.dataname, 'test', 'ir')
        self.ir_list = natsorted(os.listdir(self.ir_folder))
        print(len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)     
        # read image as type Tensor
        vis, w, h = self.imread(path=vis_path)
        ir, w, h = self.imread(path=ir_path, vis_flage=False)
        return ir.squeeze(0), vis.squeeze(0), image_name, w, h

    def __len__(self):
        return len(self.ir_list)


    @staticmethod
    def imread(path, label=False, vis_flage=True):
        if label:
            img = Image.open(path)
            width, height = img.size
            new_width = width - (width % 32)
            new_height = height - (height % 32)
            img = img.resize((new_width, new_height))
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            if vis_flage:
                img = Image.open(path).convert('RGB')
                width, height = img.size
                new_width = width - (width % 32)
                new_height = height - (height % 32)
                img = img.resize((new_width, new_height))
                im_ts = TF.to_tensor(img).unsqueeze(0)
            else:
                img = Image.open(path).convert('L')
                width, height = img.size
                new_width = width - (width % 32)
                new_height = height - (height % 32)
                img = img.resize((new_width, new_height))
                im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts, width, height
