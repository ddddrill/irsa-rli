import parametrs
import math
import cmath
import numpy as np
from  matplotlib import pyplot as plt
from scipy.fft import  ifft2, fftshift
from scipy.interpolate import griddata


class base_method():
    def __init__(self,Nf,Nph,f_r,ph_r,k_r,Nifft_fr,Nifft_ph,df,dph,FR,PH,f_c,intens,complex_v,exp_v,range_val):
        self.Nf=Nf,
        self.Nph=Nph ,
        self.f_r=f_r ,
        self.ph_r=ph_r ,
        self.k_r=k_r ,
        self.Nifft_fr=Nifft_fr ,
        self.Nifft_ph=Nifft_ph ,
        self.df=df,
        self.dph=dph ,
        self.FR=FR ,
        self.PH=PH
        self.f_c=f_c
        self.intens=intens
        self.range=range_val
        # vectorize
        self.complex_v=complex_v
        self.exp_v=exp_v
    
    
    def base_field(self,x,y):

        Es=np.zeros_like(np.arange(self.Nf[0]*self.Nph[0]).reshape(self.Nf[0],self.Nph[0]),dtype='complex128')
        ms_cos=np.multiply.outer(self.k_r[0], np.cos(self.ph_r[0]))
        ms_sin=np.multiply.outer(self.k_r[0] ,np.sin(self.ph_r[0]))
        # через вектора , как надо 
        for i in range (0,len(self.intens)):
            k_col=self.k_r[0][:,np.newaxis]
            ij=self.intens[i]*self.exp_v(self.complex_v(0,2*k_col*self.range+2*ms_cos*x[i]+2*ms_sin*y[i]))
            Es+=ij
            
        return Es

    
    
    def plot_field(self,Es=None,x=None,y=None ):
                plt.contourf(x, y, (abs(np.rot90(Es,2))),cmap='plasma')
                #plt.title('Данные поля обратного рассеяния в частотно-ориентированной области')
                plt.xlabel('Угол обзора, радиан')
                plt.ylabel('Частота , Гц')
                plt.colorbar()
                plt.show()
    
    
    def radio_image(self,Es=None,):
            # восстановим по оси OX 
            ifft_x=(self.Nifft_fr[0]*self.FR[0])/(self.Nf[0]) # Гц

            fz=self.Nifft_fr[0]*self.df[0] #
            dt=1/fz # 1/Гц
            T=self.Nifft_fr[0]*dt  # с
            t=np.arange(0,self.Nifft_fr[0])*dt
            xz=np.array((t*(3*(10**8)))/2) #
            xz=xz-xz[self.Nifft_fr[0]-1]/2

            # восстановим по оси 
            ifft_y=(self.Nifft_ph[0]*self.PH)/self.Nph[0]

            kz=self.Nifft_fr[0]*self.dph[0]
            dlen=1/kz 
            leng=np.array(np.arange(0,self.Nifft_ph[0])*dlen)#*math.pi)
            leng=leng-leng[self.Nifft_ph[0]-1]/2

            isar=(ifft_x*ifft_y*fftshift(ifft2(Es,s=[self.Nifft_fr[0],self.Nifft_ph[0]])))/(self.FR[0]*self.PH)

            return isar, xz, leng

    
    def plot_base_img(self,img=None,x=None, y=None,):
            
            d=((3*(10**8))/(2*self.f_c))

            plt.imshow(((abs(img))),extent=[x[0],x[-1],y[0]*d,y[-1]*d],cmap='plasma')
            #plt.title(f'Двумерное изображение без полярного переформатиррования\n  ширина спектра {spectr_w/10**9} ГГц')
            plt.xlabel('X , м')
            plt.ylabel('Y , м')
            plt.colorbar()
            #plt.grid()
            # plt.ylim([-2,2])
            # plt.xlim([-2,2])
            plt.show()


class polar_method():
    def __init__(self,Nf , Nph , kx , ky , kxMax, kyMax, kxMin, kyMin, Nifft_fr, Nifft_ph, M) :
        self.Nf=Nf
        self.Nph=Nph
        self.kx=kx
        self.ky=ky
        self.kxMax=kxMax
        self.kyMax= kyMax
        self.kxMin=kxMin
        self.kyMin=kyMin
        self.Nifft_fr=Nifft_fr
        self.Nifft_ph=Nifft_ph
        self.M=M

    
    def polar_field(self,Es):
        # с полярным переформатированием
        xs=complex(0,self.Nf*self.M) # создадим некое увеличение размерной сетки
        ys=complex(0,self.Nph*self.M) # для того чтоб mgrid работала нужно чтоб были комплексные числа 
        grid_x, grid_y = np.mgrid[self.kxMin:self.kxMax:xs, self.kyMin:self.kyMax:ys]
        k_x=np.ravel(self.kx) # расскладываем матрицы в вектор
        k_y=np.ravel(self.ky)
        ee=np.ravel(Es) #
        po=np.column_stack([k_x, k_y]) # складываем 1d столбцы в 2d массив
        Es_pol = griddata(po, ee, (grid_x, grid_y), method='linear', fill_value=0,)
        return Es_pol,grid_x,grid_y
    
    
    def plot_field(self,Es=None,x=None,y=None ):
                plt.contourf(x,y, (abs(Es)),cmap='plasma')
                #plt.title('Данные поля обратного рассеяния в частотно-ориентированной области')
                plt.xlabel('Ky, рад/м')
                plt.ylabel('Kx , рад/м')
                plt.colorbar()
                plt.show()

    
    def polar_image(self,Es=None,grid_x=None,grid_y=None):
        # восстановим по оси OX 
        K_x=np.amax(grid_x)-np.amin(grid_x)
        dkx=grid_x[1][0]-grid_x[0][0]

        ifft_x_p=(self.Nifft_fr*K_x)/(self.Nf*self.M)
            
        kz_xp=self.Nifft_fr*dkx
        dlen_xp=np.pi/kz_xp 
        len_xp=np.array(np.arange(0,self.Nifft_fr)*dlen_xp)#*math.pi)
        len_xp=len_xp-len_xp[self.Nifft_fr-1]/2

        # восстановим по оси OY
        K_y=np.amax(grid_y)-np.amin(grid_y)
        dky=grid_y[0][1]-grid_y[0][0]

        ifft_y_p=(self.Nifft_fr*K_y)/(self.Nph*self.M*np.pi)

        kz_yp=self.Nifft_fr*dky
        dlen_yp=np.pi/kz_yp 
        len_yp=np.array(np.arange(0,self.Nifft_ph)*dlen_yp)#*math.pi)
        len_yp=len_yp-len_yp[self.Nifft_ph-1]/2

        isar_p=(ifft_x_p*ifft_y_p)*(fftshift(ifft2(Es,s=[self.Nifft_fr,self.Nifft_ph])))  

        return isar_p, K_x ,K_y ,len_xp , len_yp      
    
    
    def plot_polar_img( self, img=None,
                       K_x=None, K_y=None,
                       len_yp=None, len_xp=None):
            plt.imshow(abs(img/(K_x*K_y)), extent=[len_yp[-1],len_yp[0],len_xp[-1],len_xp[0]],cmap='plasma')
            #plt.title('двумерное изображение после полярной интреполяции')
            plt.xlabel('X , м')
            plt.ylabel('Y , м')
            plt.colorbar()
            plt.show()  
    

class integral_method():
      def __init__(self):
            pass
      



