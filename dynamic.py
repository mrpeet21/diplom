from __future__ import division, print_function

import numpy as np
np.random.seed(42)

from scipy.constants import m_p, c, e
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from IPython import display
from scipy.stats import kde

# sets the PyHEADTAIL directory etc.
try:
    from settings import *
except:
    pass

from PyHEADTAIL.particles.generators import ParticleGenerator, RF_bucket_distribution, gaussian2D
from PyHEADTAIL.trackers.longitudinal_tracking import RFSystems
from PyHEADTAIL.trackers.rf_bucket import RFBucket

class LongitudinalDynamicGeneration:
    """Класс для моделирования продольной динамики частиц в ускорителе
        Параметры:
        params (dict): Словарь с параметрами системы:
            - V (float): Напряжение ВЧ-резонатора [V]
            - phi (float): Фазовый сдвиг ВЧ-резонатора [rad]
            - len (float): Длина кольца ускорителя [m]
            - h (list): Гармоники ВЧ-резонатора
            - charge (float): Заряд частицы [e]
            - mass (float): Масса частицы [kg]
            - gamma (float): Релятивистский фактор
            - alpha (float): Коэффициент компрессии
            - p_increment (float): Приращение импульса [eV/c]
            - intensity (float): Интенсивность пучка
            - sc (float): Масштабный коэффициент"""
    def __init__(self,params):
        
        self.V = params['V']
        self.phi = params['phi']
        self.C = params['len']
        self.h = params['h']
        self.e = params['charge']
        self.m = params['mass']
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.p_increment = params['p_increment']  
        self.intensity = params['intensity']
        self.sc = params['sc']

        self.RF_init(self.V) # Инициализация ВЧ-системы
     
    def RF_init(self, V):
        """Инициализация ВЧ-системы и расчет параметров сепаратрисы.
        Args:
            V (float): Напряжение ВЧ-резонатора [V]
        """
        self.longitudinal_map = RFSystems(self.C, self.h, V, self.phi, [self.alpha], self.gamma,
                                          p_increment=self.p_increment, charge=self.e, mass=self.m)
        self.rfbucket = self.longitudinal_map.get_bucket(gamma=self.gamma)
        self.z_max = self.C/min(self.h)/2
        self.z_min = - self.z_max
        z_long_border = np.linspace(self.z_min,self.z_max,1000)
        self.dp_max = np.max(self.rfbucket.separatrix(z_long_border))
        self.dp_min = -self.dp_max
        
   
        
    def beam_init(self, sigma_z, sigma_p, max_z, max_p, c_z_u, c_z_n, c_p, alpha, macroparticlenumber): 
        """Инициализация пучка с заданными параметрами.
        Args:
            sigma_z (float): СКО распределения по z для гауссовой компоненты [m]
            sigma_p (float): СКО распределения по dp для гауссовой компоненты
            max_z (float): Максимальное отклонение по z для равномерной компоненты [m]
            max_p (float): Максимальное отклонение по dp для равномерной компоненты
            c_z_u (float): Центр равномерного распределения по z [m]
            c_z_n (float): Центр гауссова распределения по z [m]
            c_p (float): Центр распределения по dp
            alpha (float): Доля частиц с гауссовым распределением (0-1)
            macroparticlenumber (int): Количество макрочастиц
        """
        self.beam =  ParticleGenerator(
            macroparticlenumber, self.intensity, self.e, self.m, self.C, self.gamma, 
            distribution_x = gaussian2D(1e-4),
            distribution_y = gaussian2D(1e-4),
            distribution_z = self.gaus_unif_mix(sigma_z, sigma_p, max_z, max_p, c_z_u, c_z_n, c_p, alpha)).generate()
        
        # alpha =1 - gauss / alpha =0 - uniform/ 0< alpha <1 - mix
        
    def gaus_unif_mix(self,sigma_z, sigma_p, max_z, max_p, c_z_u, c_z_n, c_p, alpha):
        """Генератор смешанного (гауссово + равномерное) распределения. 
        Args:
            sigma_z (float): СКО по z для гауссовой компоненты [m]
            sigma_p (float): СКО по dp для гауссовой компоненты
            max_z (float): Максимальное отклонение по z для равномерной компоненты [m]
            max_p (float): Максимальное отклонение по dp для равномерной компоненты
            c_z_u (float): Центр равномерного распределения по z [m]
            c_z_n (float): Центр гауссова распределения по z [m]
            c_p (float): Центр распределения по dp
            alpha (float): Доля гауссовой компоненты (0-1)
       Returns:
            _mix: Функция-генератор распределения, возвращающая [z, dp]
        """
        def _mix(n_particles):

            n_particles_0 = round(alpha*n_particles) # Количество частиц с гауссовым распределением
            n_particles_1 = n_particles - round(alpha*n_particles) # Количество частиц с равномерным распределением

            # Гауссова компонента
            z_1  = np.random.normal(c_z_n, sigma_z, n_particles_0)
            dp_1 = np.random.normal(c_p, sigma_p, n_particles_0)
            
            # Равномерная компонента
            z_2 = np.random.uniform(low=-max_z+c_z_u, high=max_z+c_z_u, size = n_particles_1)
            dp_2 = np.random.uniform(low=-max_p+c_p, high=max_p+c_p, size = n_particles_1)

            # Объединение распределений
            z = np.concatenate((z_1,z_2),0)
            dp = np.concatenate((dp_1,dp_2),0)
            #print (z,dp)
            return [z, dp]
        return _mix
                        


    def dynamic(self, n_turns):
        """Моделирование динамики частиц на заданном числе оборотов.    
        Args:
            n_turns (int): Количество оборотов для моделирования        
        Returns:
            tuple: (z_array_per_turn, dp_array_per_turn, b_z_arr, b_dp_arr)
                - z_array_per_turn: Массив z-координат частиц внутри сепаратрисы на каждом обороте
                - dp_array_per_turn: Массив dp частиц внутри сепаратрисы на каждом обороте
                - b_z_arr: Массив z-координат потерянных частиц
                - b_dp_arr: Массив dp потерянных частиц
        """

        z_array_per_turn = []
        dp_array_per_turn = []
        
        b_z_arr=[]
        b_dp_arr=[]
        zmax = self.C / min(self.h) / 2
        zzsep = np.linspace(-zmax, zmax, 1000)
        
        for i in range(n_turns):
        # Обновление ВЧ-напряжения и трекинг
            self.longitudinal_map.voltages = self.V   
            rfbucket = self.longitudinal_map.get_bucket(self.beam)
            self.longitudinal_map.track(self.beam)
            
        # Частицы внутри сепаратрисы
            mskdp1 = (self.beam.dp < rfbucket.separatrix(self.beam.z)) & (self.beam.dp > -rfbucket.separatrix(self.beam.z))
            ind1 = np.where((mskdp1 == True))
            mskdp2 = (self.beam.z[ind1] < zmax) & (-zmax < self.beam.z[ind1])
            ind2 = np.where((mskdp2 == True))
            
        # Частицы вне сепаратрисы (потерянные) 
            mskdp1 = (self.beam.dp < rfbucket.separatrix(self.beam.z)) & (self.beam.dp > -rfbucket.separatrix(self.beam.z))
            bind1=np.where((mskdp1 == False))
            b_mskdp2 = (self.beam.z[bind1] < zmax) & (-zmax < self.beam.z[bind1])
            bind2 = np.where((b_mskdp2 == False))
            
          # Сохранение результатов
            b_z_arr.append(self.beam.z[bind1][bind2])
            b_dp_arr.append(self.beam.dp[bind1][bind2])
            
            z_array_per_turn.append(self.beam.z[ind1][ind2])
            dp_array_per_turn.append(self.beam.dp[ind1][ind2])

                
    
        return  z_array_per_turn, dp_array_per_turn, b_z_arr, b_dp_arr
    
    # набросок для вывода распределения
    def plot_r(self, z, dp,color="red"):
        """Визуализация распределения частиц в фазовой плоскости.  
        Args:
            z (array): Массив z-координат частиц
            dp (array): Массив dp частиц
            color (str): Цвет точек на графике
        """
        zmax = self.C/min(self.h)/2.
        zzsep = np.linspace(-zmax, zmax, 1000)
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(20)
        ax.scatter(z, dp, color=color, label = r'$\epsilon_x$')
        ax.plot(zzsep, self.rfbucket.separatrix(zzsep))
        ax.plot(zzsep, -self.rfbucket.separatrix(zzsep))
        ax.legend(loc='upper left')
        ax.minorticks_on()
        ax.grid(which='major',color = 'k', linewidth = 2)
        ax.grid(which='minor', color = 'k', linestyle = ':')
        #ax.set_xlim(-14, 14)
        #ax.set_ylim(-2e-3,2e-3)
       
        plt.show()
    # эта функция нигде не используется   
    def plot_gist(self, n_turns, z, dp):
        kde = kde.gaussian_kde([z,dp])
        #отчего мы строим гистограмму 
        hh = self.rfbucket.hamiltonian(z, dp)
        zi, dpi = np.mgrid[z.min():dp.max():n_turns*1j, z.min():dp.max():n_turns*1j]
        Z_i = kde(np.vstack([zi.flatten(), dpi.flatten()]))
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        axes.pcolormesh(zi, dpi, Z_i.reshape(zi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
        return  
    
    def data_preparation(self, n_turns, N_input, N_output, z, dp): 
        """Подготовка данных для обучения нейросети. 
        Args:
            n_turns (int): Количество оборотов
            N_input (int): Размер входного изображения
            N_output (int): Размер выходного изображения
            z (list): Массив z-координат по оборотам
            dp (list): Массив dp по оборотам
        Returns:
            tuple: (gists, I)
                - gists: Гистограммы распределения частиц по z
                - I: 2D распределение частиц в фазовой плоскости (z, dp)
        """
        l_low = -self.C / min(self.h) / 2.
        l_high = self.C / min(self.h) / 2.
        n_bins = N_input
        delta_a= (l_high-l_low)/n_bins
        gists_per_turns = np.zeros((n_turns,n_bins))
        step = n_turns//N_input 

        for k in range(0,n_turns):
            position = z[k]
            for i in range(position.shape[0]):
                for j in range(n_bins):
                    if (l_low + j*delta_a <= position[i]) and (l_low + (j+1)*delta_a > position[i]):
                        gists_per_turns[k,j]+=1
        gists = gists_per_turns[::step,:]
        gists = gists[0:N_input,:]
        x, y = z[0], dp[0]
        nbins = N_output 
        k = gaussian_kde([x, y])
        xi, yi = np.mgrid[self.z_min*self.sc:self.z_max*self.sc:nbins*1j, self.dp_min*self.sc:self.dp_max*self.sc:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        I = zi.reshape(xi.shape) 
        I = np.rot90(I)
               
        return gists, I
    
    def data_preparation_picture(self, n_turns, N_input, N_output, z, dp):
        """Визуализация подготовленных данных. 
        Args:
            n_turns (int): Количество оборотов
            N_input (int): Размер входного изображения
            N_output (int): Размер выходного изображения
            z (list): Массив z-координат
            dp (list): Массив dp
        """
        x , y = self.data_preparation(n_turns, N_input, N_output, z, dp)
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(20)
        ax.imshow(y, cmap=plt.cm.BuGn_r)
        plt.show() 
        
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(20)
        ax.imshow(x, cmap=plt.cm.BuGn_r)
        plt.show()
