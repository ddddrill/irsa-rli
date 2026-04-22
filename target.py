import math
import numpy as np
import os
from  matplotlib import pyplot as plt

class Target:

    def __init__ (self,file,plot_flag=True):
            self.file=file
            self.plot=plot_flag

    def targets_matrix(self):
        '''
        by reading the file, the program receives data on where the main scattering centers are located (takes the maximum intensity from the file)
        file: data 
        '''
        # это все данные которые есть  
        detail_path=os.path.dirname(os.path.abspath(__file__))+'\\'+self.file
        # print (detail_path)
        raw_s = r'{}'.format(detail_path)
        detail_imgs=np.load(raw_s,allow_pickle=True)

        return detail_imgs
    
    def single_target(self,n):
        dots=Target.target_x_y() 
        curr=dots[n]
        return curr
