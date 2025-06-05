import numpy as np
import torch
import matplotlib.pyplot as plt

import math
from unet import UNet
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook, trange, tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from dynamic import LongitudinalDynamicGeneration
from scipy.constants import m_p, c, e
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class CustomDataset(Dataset):
    def __init__(self, params, params_gen):
        
        self.__len_dataset__ = params['len_dataset']
        self.sigma_z = params['sigma_z']
        self.sigma_p = params['sigma_p']
        self.max_z = params['max_z']
        self.max_p = params['max_p']
        self.c_z_u = params['c_z_u']
        self.c_z_n = params['c_z_n']
        self.c_p = params['c_p']
        self.macroparticlenumber = params['macroparticlenumber']
        self.N_turns = params['N_turns']
        self.N_input = params['N_input']
        self.N_output = params['N_output']
        self.V = params['V']
        self.generator = LongitudinalDynamicGeneration(params_gen)
        
    def beam_initialization(self, alpha, sigma_z, sigma_p, max_z, max_p, c_z_u, c_z_n, c_p):

        self.generator.beam_init(sigma_z, sigma_p, max_z, max_p, c_z_u, c_z_n, c_p, alpha, self.macroparticlenumber)
        
    def __len__(self):
        return self.__len_dataset__

    def __getitem__(self, idx):

        V = []
        for i in range(len(self.V)):
        
            V.append(self.V[i][0] + np.random.random()*(self.V[i][1]-self.V[i][0]))
        self.generator.RF_init(V)
        # для равномерного 
        stop = False
        while not stop:
            max_z = self.max_z[0]+np.random.random()*(self.max_z[1]-self.max_z[0])
            c_z_u = self.c_z_u[0]+np.random.random()*(self.c_z_u[1]-self.c_z_u[0])
            if c_z_u <= -0.7*max_z + 10.4 and max_z <= -10/7*c_z_u +14.85:
                stop = True
                
        # для нормального 
        stop = False 
        while not stop:
            sigma_z = self.sigma_z[0]+np.random.random()*(self.sigma_z[1]-self.sigma_z[0])
            c_z_n = self.c_z_n[0]+np.random.random()*(self.c_z_n[1]-self.c_z_n[0])
            if c_z_n <= -sigma_z + 8 and sigma_z <= -1*c_z_n + 8:
                stop = True
            
        sigma_p = self.sigma_p[0]+np.random.random()*(self.sigma_p[1]-self.sigma_p[0])
        
        max_p = self.max_p[0]+np.random.random()*(self.max_p[1]-self.max_p[0])
        c_p = self.c_p[0]+np.random.random()*(self.c_p[1]-self.c_p[0])
        alpha = np.random.rand()
        self.beam_initialization(alpha, sigma_z, sigma_p, max_z, max_p, c_z_u, c_z_n, c_p)
        z, dp, _, _ = self.generator.dynamic(self.N_turns)
        plt.scatter(z[0], dp[0])
        gists, I = self.generator.data_preparation(self.N_turns, self.N_input, self.N_output, z, dp)
        gists = gists[None,:,:]
        I = I[None,:,:]
        return gists.astype(np.float32), I.astype(np.float32)

class Training():
    def __init__(self, params_model, params_training):
        self.device = params_training['device']
        self.flag_load_model = params_training['flag_load_model']
        self.name_for_save = params_training['name_for_save']
        self.name_log_loss = params_training['name_log_loss']
        self.model = UNet(in_channels=1, out_channels=1)
        if self.flag_load_model:
            self.model.load_state_dict(torch.load(self.name_for_save, map_location = 'cpu'))
        self.model.to(self.device)
        self.loss_function = nn.MSELoss()
        self.epoch = 0
        try:
            os.remove(self.name_log_loss)
        except: 
            pass
        
    def fit(self, train_loader, epochs,  optimizer):
   
        self.model.train()
        
        for epoch in range(epochs):
            self.epoch += 1 
            loss_val = 0 
            nmse_val = 0
            k = 0
            for x_input , x_output in tqdm(train_loader, desc='Training', position=0, leave=True):
                x_input = x_input.to(self.device)
                x_output = x_output.to(self.device)
                
                prediction = self.model(x_input)
                
                loss  = self.loss_function(prediction, x_output)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_val = loss_val + loss.item()
                k = k + 1
                if np.isnan(loss_val):
                    break
            loss_val = loss_val / k
            print('epoch = ' + str(epoch) +  '  loss = ' + str(np.round(loss_val, 4)) )

            self.log_loss(loss_val)
            
            if np.isnan(loss_val) == False:
                # узнать правильно это или нет
                #current_date = datetime.now().strftime('%Y-%m-%d')
                if self.epoch % 50 == 0 and self.epoch!= 0:
                    save_path = f"{self.name_for_save}_{self.epoch}.pt"
                    torch.save(self.model.state_dict(), save_path)

    def train_one_epoch(self, train_loader, optimizer):
        self.model.train()
        loss_val = 0
        k = 0

        for x_input , x_output in tqdm(train_loader, desc=f'Epoch {self.epoch}', position=0, leave=True):
            x_input = x_input.to(self.device)
            x_output = x_output.to(self.device)

            prediction = self.model(x_input)
            loss = self.loss_function(prediction, x_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
            k += 1

        loss_val /= k
        print(f'Epoch {self.epoch} | Loss: {loss_val:.6f}')
        self.log_loss(loss_val)
        self.epoch += 1
        if np.isnan(loss_val) == False:
                # узнать правильно это или нет
                #current_date = datetime.now().strftime('%Y-%m-%d')
            if self.epoch % 50 == 0 and self.epoch!= 0:
                save_path = f"{self.name_for_save}_{self.epoch}.pt"
                torch.save(self.model.state_dict(), save_path)
        return loss_val
        
    
    def result(self, train_data_loader):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        input_, output_ = next(iter(train_data_loader))
        output_ = output_.cpu().numpy()
        with torch.no_grad():
            pred = self.model(input_.to(self.device))
            pred = pred.detach().cpu().numpy()
        input_ = input_.cpu().numpy()
        axes[0].imshow(input_[0,0].squeeze())
        axes[0].axis("off")
        axes[0].set_title(f"Входные данные")
            
        
        axes[1].imshow(pred[0,0].squeeze())
        axes[1].axis("off")
        axes[1].set_title(f"Нейросеть")
            
        
        axes[2].imshow(output_[0,0].squeeze())
        axes[2].axis("off")
        axes[2].set_title(f"Генератор")
        
        plt.tight_layout()
        plt.show()
        
    def log_loss(self, loss_val):
        with open(self.name_log_loss, "a") as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Loss: {loss_val}\n")


        
if __name__ == "__main__":
    
    params_dataset = {}
    params_dataset['V'] = [[90e3, 100e3]]
    params_dataset['len_dataset'] = 512
    params_dataset['sigma_z'] = [10/5, 10/2] #границы для sigma_z
    params_dataset['sigma_p'] = [0.001, 0.0016]
    params_dataset['max_z'] = [2, 12]
    params_dataset['max_p'] = [1e-5, 1e-3]
    params_dataset['c_z_u'] = [0, 9] #параметр для равноменого
    params_dataset['c_z_n'] = [0, 6] #параметр для нормального 
    params_dataset['c_p'] = [0, 7e-4]
    params_dataset['alpha'] = 0.5
    params_dataset['macroparticlenumber'] = 2000
    params_dataset['N_turns'] = 1000
    params_dataset['N_input'] = 128
    params_dataset['N_output'] = 128   
    
    p0 = 1.4e9 * e/c
    gamma  = np.sqrt(1 + (p0/(m_p*c))**2)
    beta = np.sqrt(1-1/gamma**2)
    dp = 0. * e/p0 * 1./(beta*c) #* normalisation
    params = {}
    params['len']= 504
    params['phi'] = [np.pi]
    params['V'] = [80e3]
    params['h'] = [22]
    params['gamma'] = np.sqrt(1 + (p0/(m_p*c))**2)
    params['charge'] = e
    params['mass'] = m_p
    params['alpha'] = 0.027 
    params['p_increment'] = 0 #? dp*p0
    params['intensity'] = 1e10
    params['sc'] = 1.05
    
    params_model = {}
    params_model['in_nc'] = 1
    params_model['nf'] = 100
    params_model['num_modules'] = 4
    params_model['out_nc'] = 1
    params_model['upscale'] = 1
    params_training = {}
    params_training['device'] = 'cuda:0'
    params_training['flag_load_model'] = False
    params_training['name_for_save'] = 'ckpt_400_2/model_UNET'
    params_training['name_log_loss'] = 'log_loss_400_2'
    
    batch_size = 16
    
    dset = CustomDataset(params_dataset, params)
    train_data_loader = DataLoader(dset, batch_size = batch_size, shuffle = True, num_workers = 24)
    
    train = Training(params_model, params_training)
    
    count_parameters(train.model)
    
    optimizer = torch.optim.AdamW([{'params': train.model.parameters(), 'lr': 5e-4}])
    train.fit(train_data_loader, 100, optimizer)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    #train.model.load_state_dict(torch.load('ckpt_400/model_UNET_150.pt', weights_only=False))
    #train.model.eval()
    
    optimizer = torch.optim.AdamW([{'params': train.model.parameters(), 'lr': 1e-4}])
    train.fit(train_data_loader, 100, optimizer)
    
    
    optimizer = torch.optim.AdamW([{'params': train.model.parameters(), 'lr': 5e-5}])
    train.fit(train_data_loader, 100, optimizer)

    optimizer = torch.optim.AdamW([{'params': train.model.parameters(), 'lr': 1e-5}])
    train.fit(train_data_loader, 100, optimizer)

    #optimizer = torch.optim.AdamW([{'params': train.model.parameters(), 'lr': 1e-4}])
    

    """optimizer = torch.optim.AdamW([{'params': train.model.parameters(), 'lr': 5e-5}])
    train.fit(train_data_loader, 100, optimizer)

    optimizer = torch.optim.AdamW([{'params': train.model.parameters(), 'lr': 1e-5}])
    train.fit(train_data_loader, 100, optimizer)"""
    
