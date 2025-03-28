import torch
import torch.nn as nn
import torchvision.transforms as TF
from torchvision.utils import save_image
import cv2
import numpy as np
import random
import kornia.geometry.transform as KT
import matplotlib.pyplot as plt
import random
from torchvision import models
from PIL import Image


class PatchTransformer(nn.Module):
    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.noise_factor = 0.1
    


    def forward(self, patch, targets, imgs):
        # patch = patch.unsqueeze(0)
        # patch = patch.expand(-1, 3, -1, -1)
        # patch = np.zeros((3, 3), dtype=np.float32)  #
        # patch = torch.from_numpy(patch)
        # patch = torch.ones_like(patch)
        patch_mask = torch.ones_like(patch).cuda()
        image_size = imgs.size()         # height, width
        #print(targets.size())git 
        #print(targets[:, 0])
        #print(image_size)
        
        #patch_tmp = torch.zeros([1, 3, image_size[-2], image_size[-1]], dtype=torch.uint8).cuda()
        #patch_mask_tmp = torch.zeros([1, 3, image_size[-2], image_size[-1]], dtype=torch.uint8).cuda()
        patch_tmp = torch.zeros_like(imgs).cuda()
        patch_mask_tmp = torch.zeros_like(imgs).cuda()
        # print('---4--->', patch.size())
        
        for i in range(targets.size(0)):
            img_idx = targets[i][0]
            # print(i)
            # print(patch.data.device)
            # print(torch.cuda.get_device_name(0))
            ''' 
            # noise
            noise = torch.cuda.FloatTensor(patch.size()).uniform_(-1, 1) * self.noise_factor
            patch = patch + noise
            # patch = torch.clamp(patch, 0.000001, 0.99999)
            patch.data.clamp_(0,1)
            '''
            
            # resize
            patch_size = int(targets[i][-1] * image_size[-2] * 1) # 0.7
            # print('-------0622----->', patch_size)
            patch_resize = KT.resize(patch, (patch_size, patch_size), align_corners=True)
            patch_mask_resize = KT.resize(patch_mask, (patch_size, patch_size), align_corners=True)
            # print("-----patch_resize---->", patch_resize.requires_grad)
            
            # print('------>', patch_resize.size())
            # rotation
            angle = random.uniform(0, 0)
            patch_rotation = TF.functional.rotate(patch_resize, angle, expand=True)
            patch_mask_rotation = TF.functional.rotate(patch_mask_resize, angle, expand=True)
            
            patch_size_h = patch_rotation.size()[-1]
            patch_size_w = patch_rotation.size()[-2]
            
            # padding
            x_center = int(targets[i][2] * image_size[-1])
            y_center = int(targets[i][3] * image_size[-2])
            
            padding_h = image_size[-2] - patch_size_h
            padding_w = image_size[-1] - patch_size_w
            
            padding_left = x_center - int(0.5 * patch_size_w)
            padding_right = padding_w - padding_left
            
            padding_top = y_center - int(0.6 * patch_size_h)
            padding_bottom = padding_h - padding_top

            # print('----3---->', patch_rotation.size())

            padding = nn.ZeroPad2d((int(padding_left), int(padding_right), int(padding_top), int(padding_bottom)))
            patch_padding = padding(patch_rotation)
            patch_mask_padding = padding(patch_mask_rotation)
            #print("-----patch_padding---->", patch_padding.requires_grad)
            
            # print('----1---->', patch_padding.size())
            # print('----2---->', patch_tmp.size())
            patch_tmp[int(img_idx.item())] += patch_padding.squeeze()
            patch_mask_tmp[int(img_idx.item())] += patch_mask_padding.squeeze()
            
        #patch_tf = torch.cat(patch_list, 0)
        #patch_mask_tf = torch.cat(patch_mask_list, 0)
        #patch_tf = torch.clamp(patch_tmp, 0.000001, 0.99999)
        #patch_mask_tf = torch.clamp(patch_mask_tmp, 0, 1)
        patch_tmp.data.clamp_(0,1)
        # patch_mask_tmp.data.clamp_(0,1)
        #print("-----patch_tf---->", patch_tf.requires_grad)
        '''
        # 保存中间图片
        img_save = patch_tf[0]
        im = TF.ToPILImage()(img_save)
        im.save("1.png")
        img_save = patch_mask_tf[0]
        im = TF.ToPILImage()(img_save)
        im.save("2.png")
        print(patch_tf.size())
        '''
        return patch_tmp, patch_mask_tmp
        
        
        
class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, patch, patch_mask_tf):
        patch_mask = patch_mask_tf - 1
        patch_mask = - patch_mask

        #img_batch = torch.mul(img_batch, patch_mask) + torch.mul(patch, patch_mask_tf)
        img_batch = patch * 0.7 + img_batch * 1.0;
        img_batch = img_batch.data.clamp_(0,1)
        #img_batch = torch.mul(img_batch, torch.ones_like(img_batch))
        
        '''
        img_save = img_batch[3]
        im = TF.ToPILImage()(img_save)
        im.save("3.png")
        img_save = img_batch[2]
        im = TF.ToPILImage()(img_save)
        im.save("5.png")
        img_save = img_batch[1]
        im = TF.ToPILImage()(img_save)
        im.save("6.png")
        img_save = img_batch[0]
        im = TF.ToPILImage()(img_save)
        im.save("4.png")
        #print(img_save)
        '''
        
        imgWithPatch = img_batch
        return imgWithPatch


# define draw
def plotCurve(x_vals, y_vals, 
                x_label, y_label, filename,
                legend=None,
                figsize=(10.0, 5.0)):
    # set figsize
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    
    if legend:
        plt.legend(legend)
    plt.savefig('results/'+filename)
    #plt.show()
    plt.close('all')

       
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            #print('t', t)
            #print('m', m)
            #print('s', s)
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
        
        
def image_loader(image_name, size):
    loader = TF.Compose([
          TF.Resize(size),  # scale imported image
          TF.CenterCrop(size),
          TF.ToTensor(),
          TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
          # TF.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
          ])  # transform it into a torch tensor
    
    image = Image.open(image_name).convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out
        
        
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G
    

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename).convert('RGB')
    return img


# using ImageNet values
def normalize_tensor_transform():
    return TF.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        
        
        
        
        
        
        
        
        
        
        