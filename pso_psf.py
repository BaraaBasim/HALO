import copy
import time
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision.transforms as TF
from utils import PatchApplier, PatchTransformer
from script import generatePSF, add_padding
import matlab.engine


class OptimizeFunction:
    def __init__(self, detector, device):
        self.detector = detector
        self.device = device
        self.trans = PatchTransformer()
        self.patch_applier = PatchApplier()

    def set_para(self, targets, imgs):
        self.targets = targets
        self.imgs = imgs
        # self.eng = eng

    def evaluate(self, x):
        # print(x)
        #x.data.clamp_(0,700)
        with torch.no_grad():
            imgWithPatch = self.imgs 
            # Number of Light sources 12
            for i in range(12):
                top_para, bottom_para, left_para, right_para = x[i][0].item(), x[i][1].item(), x[i][2].item(), x[i][3].item();
                
                psf_img = cv2.imread(f'psf_starlight/{i}.jpg')
                psf_img = cv2.cvtColor(psf_img, cv2.COLOR_BGR2RGB)

                
                im = TF.ToPILImage()(psf_img)
                im.save("psf_before_padding.png")


                psf_img = add_padding(psf_img, top_para, bottom_para, left_para, right_para)
                
                
                im = TF.ToPILImage()(psf_img)
                im.save("psf_after_padding.png")
                
                # ---------------------------
                # print('---top_para--->', top_para)
                # print('---bottom_para--->', bottom_para)
                # print('---left_para--->', left_para)
                # print('---right_para--->', right_para)
                
                # print('---color_parar--->', color_paraR)
                # print('---color_parag--->', color_paraG)
                # print('---color_parab--->', color_paraB)


                psf_img = psf_img.astype(np.uint8)
                psf_img = psf_img / 255
                psf_img = torch.from_numpy(psf_img).cuda()
                psf_img = psf_img.permute(2, 0, 1)

                # ---------------------------
                # im = TF.ToPILImage()(psf_img)
                # im.save("1_psf.png")
                
                # ---------------------------

                patch_tf, patch_mask_tf = self.trans(psf_img, self.targets, self.imgs)
                imgWithPatch = self.patch_applier(imgWithPatch, patch_tf, patch_mask_tf)

            # ---------------------------
            img_save = imgWithPatch[0]
            im = TF.ToPILImage()(img_save)
            im.save("0.png")
            # # ---------------------------

            out, train_out = self.detector(imgWithPatch)
            obj_confidence = out[:, :, 4]
            max_obj_confidence, _ = torch.max(obj_confidence, dim=1)
            obj_loss = torch.mean(max_obj_confidence)
            
            return_obj_loss = obj_loss
            plt.plot(return_obj_loss.cpu().numpy(), label="Training Loss")
                # Loss function
        # plt.show()
        return return_obj_loss



class SwarmParameters:
    pass


class Particle:
    def __init__(self, dimensions, device):
        self.device = device
        self.dimensions = dimensions
        self.w = 0.5
        self.c1 = 2
        self.c2 = 2
        self.classes = 4
        
        random_matrix = torch.rand((12, 4)).to(self.device) * 700
        
        self.position = random_matrix
        self.velocity = torch.zeros((dimensions, self.classes)).to(self.device)
        
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)
        
    
    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        for i in range(0, self.classes):
            self.velocity[i] = self.w * self.velocity[i] \
                               + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                               + self.c2 * r2 * (gbest_position[i] - self.position[i])

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters
        
        
    def move(self):
        for i in range(0, self.dimensions):
            self.position[i] = self.position[i] + self.velocity[i]
            
        # random_matrix = torch.rand((4, 4)).to(self.device)
        # random_matrix[:, :] = random_matrix[:, :] * 700
        # # random_matrix[0, 3:6] = random_matrix[0, 3:6] * 256

        # self.position = random_matrix
        # self.position.data.clamp_(0,700)
        self.position.data.clamp_(0,700)
        

class PSO:
    def __init__(self, swarm_size, device):
        self.max_iterations = 3
        self.swarm_size = swarm_size
        self.gbest_position = [0, 0]
        self.gbest_particle = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)
        self.swarm = []
        for i in range(self.swarm_size):
            self.swarm.append(Particle(dimensions=12, device=device))         # dimension
        
    
    def optimize(self, function):
        self.fitness_function = function
        
        
    def run(self):
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0
        # --- Run
        for iteration in range(self.max_iterations):
            # --- Set PBest
            for particle in self.swarm:
                fitness_candidate = self.fitness_function.evaluate(particle.position)
                #break
                if (particle.pbest_value > fitness_candidate):
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position = particle.position.clone()
            # --- Set GBest
            for particle in self.swarm:
                best_fitness_candidate = self.fitness_function.evaluate(particle.position)
                if self.gbest_value > best_fitness_candidate:
                    self.gbest_value = best_fitness_candidate
                    self.gbest_position = particle.position.clone()
                    self.gbest_particle = copy.deepcopy(particle)
            
            r1s = []
            r2s = []
            # --- For Each Particle Update Velocity
            for particle in self.swarm:
                parameters = particle.update_velocity(self.gbest_position)
                particle.move()
                r1s.append(parameters.r1)
                r2s.append(parameters.r2)
            
            swarm_parameters.r1 = (sum(r1s) / self.swarm_size).item()
            swarm_parameters.r2 = (sum(r2s) / self.swarm_size).item()


        swarm_parameters.gbest_position = self.gbest_position
        swarm_parameters.gbest_value = self.gbest_value.item()
        swarm_parameters.c1 = self.gbest_particle.c1
        swarm_parameters.c2 = self.gbest_particle.c2
        return swarm_parameters
        