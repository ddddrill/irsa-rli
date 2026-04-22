import math
import cmath
import numpy as np


# params
class Parametrs():

    def __init__(self,c:float,Xmax:float,Ymax:float,
                 f_c:int, spectr_w:int, ph_c:int,
                 ang:float, nifft_size:int=1024):
        # define
        self.c=c # скорость света
        self.Xmax=Xmax # задаем размер окна с которого мы будем снимать изображение 2D по оси OX
        self.Ymax=Ymax # по OY
        self.f_c=f_c  # ценральная частота
        self.k_c=(2*math.pi*f_c)/self.c #  Центральное волновое число
        self.spectr_w=spectr_w # ширина спектраширина обзора в радианах (но подают градусы тк удобство)
        self.ph_c=ph_c # центральный угол (берем за ноль)
        self.ang=ang # угол обзора в градусах
        self.nifft_size = nifft_size

        # calculate
        
        self.ph_w=math.radians(self.ang) #  переводим в радианы угол обзора
        self.resolution_x=self.c/(self.spectr_w*2) # разрешение по OY
        self.resolution_y=(math.pi/self.k_c)/(self.ph_w) # разрешение по OX
        self.Nf = math.ceil((self.Xmax*self.spectr_w*2)/self.c) #число отсчетов по прямой
        self.Nph = math.ceil((2*Ymax*self.ph_w*f_c)/self.c)
            # Пространственные вектора 
        X=np.arange(-self.resolution_x*(self.Nf/2), self.resolution_x*(self.Nf/2), self.resolution_x)
        Y=np.arange(-self.resolution_y*(self.Nph/2), self.resolution_y*(self.Nph/2), self.resolution_y)

        self.df=self.spectr_w/self.Nf # разрешение по частоте
        self.dph=self.ph_w/self.Nph # разрешение по ширине обзора


        self.cos=np.vectorize(math.cos) # векторизируем функцию
        self.sin=np.vectorize(math.sin) # векторизируем функцию
        self.exp=np.vectorize(cmath.exp) # векторизируем функцию
        self.complex_v=np.vectorize(complex) # векторизируем функцию
        self.floor=np.vectorize(math.floor)

    def vector_func(self):
        return self.cos, self.sin, self.exp, self.complex_v, self.floor
    
    def base_img_param(self,):
        # field
        self.Nf = math.ceil((self.Xmax*self.spectr_w*2)/self.c) #число отсчетов по прямой
        self.Nph = math.ceil((2*self.Ymax*self.ph_w*self.f_c)/self.c)
        self.f_r = np.linspace(self.f_c-self.df*(self.Nf/2), self.f_c+self.df*(self.Nf/2)-self.df, self.Nf) # вектор частот
        self.ph_r = np.linspace(self.ph_c-self.dph*(self.Nph/2), self.ph_c+self.dph*(self.Nph/2)-self.dph, self.Nph) # вектор углов
        self.k_r = 2*math.pi*self.f_r/self.c # вектор волнового числа
        # image
        self.Nifft_fr=self.nifft_size
        self.Nifft_ph=self.nifft_size       
        self.df=self.spectr_w/self.Nf # разрешение по частоте
        self.dph=self.ph_w/self.Nph # разрешение по ширине обзора 
        self.FR = max(self.f_r) - min(self.f_r) # полоса пропускания по частоте
        self.PH = max(self.ph_r) - min(self.ph_r) # полоса пропускания по углу обзора

        return self.Nf , self.Nph , self.f_r , self.ph_r , self.k_r , self.Nifft_fr , self.Nifft_ph , self.df, self.dph , self.FR , self.PH
   
    def polar_img_param(self,):
        # field
        self.f_r = np.linspace(self.f_c-self.df*(self.Nf/2), self.f_c+self.df*(self.Nf/2)-self.df, self.Nf)
        self.ph_r = np.linspace(self.ph_c-self.dph*(self.Nph/2), self.ph_c+self.dph*(self.Nph/2)-self.dph, self.Nph)
        self.k_r = 2*math.pi*self.f_r/self.c
        self.Nf = math.ceil((self.Xmax*self.spectr_w*2)/self.c) #число отсчетов по прямой
        self.Nph = math.ceil((2*self.Ymax*self.ph_w*self.f_c)/self.c)
        self.kx=np.outer(self.k_r,self.cos(self.ph_r)) # матрица размером Nf и Nph
        self.ky=np.outer(self.k_r,self.sin(self.ph_r)) # аналогично
        self.kxMax = np.amax(self.kx) # выделяет максимум разложенной матрицы 
        self.kxMin = np.amin(self.kx) # 
        self.kyMax = np.amax(self.ky) # 
        self.kyMin = np.amin(self.ky)
        # img
        self.Nifft_fr=self.nifft_size
        self.Nifft_ph=self.nifft_size  
        self.M=4

        return  self.Nf, self.Nph, self.kx, self.ky, self.kxMax, self.kyMax, self.kxMin, self.kyMin, self.Nifft_fr, self.Nifft_ph, self.M
    
    def integral_param(self,):
        # field
        resolution_x = self.c/(self.spectr_w*2) # разрешение по OY
        resolution_y=(math.pi/self.k_c)/(self.ph_w) # разрешение по OX
        X=np.arange(-resolution_x*(self.Nf/2), resolution_x*(self.Nf/2), resolution_x)
        Y=np.arange(-resolution_y*(self.Nph/2), resolution_y*(self.Nph/2), resolution_y)
        X_int=np.arange(min(X),max(X),0.05)
        Y_int=np.arange(min(Y),max(Y),0.05)
        Es_int=np.zeros((X_int.size,Y_int.size))

        h_k = self.dk    #(kMax-kMin)/(Nf)
        k1=np.linspace(min(self.k_r),max(self.k_r),self.Nf)
        wk1=np.zeros((k1.size,k1.size), dtype=int)
        wk1[1:((self.Nf+1)+1):2]=4
        wk1[2:((self.Nf+1)+1):2]=2
        wk1[0]=1
        wk1[-1]=1
        #wk1[:,0]
        wk1=wk1[:,0]*(h_k/3)

        h_ph =self.dph
        ph1=np.linspace(min(self.ph_r),max(self.ph_r),(self.Nph))
        wph1=np.zeros((ph1.size,ph1.size), dtype=int)
        wph1[1:((self.Nph+1)+1):2]=4
        wph1[2:((self.Nph+1)+1):2]=2
        wph1[0]=1
        wph1[-1]=1
        wph1=wph1[:,0]*(h_ph/3)

        #создаем матрицы отталкиваясь от векторов частоты 
        ph1, k1 = np.meshgrid(ph1, k1)
        # переводим матрицу в вектор
        ph1=np.ravel(ph1)
        k1=np.ravel(k1)
        # комапануем веса функций для дальнейшего интегрирования
        w=np.outer(wk1, wph1)
        w_ravel=(np.ravel(np.outer(wk1, wph1)))
        Es_rr=np.ravel(np.zeros_like(np.arange(self.Nf*self.Nph).reshape(self.Nf,self.Nph)))
        new_w=w_ravel*Es_rr
        #plt.hist(w_ravel.reshape((w_ravel.size),1))
        new_w=new_w.reshape((1,w_ravel.size))
        k_ph_x=k1*self.cos(ph1)
        k_ph_y=k1*self.sin(ph1)
    
        return X_int,Y_int, k_ph_x, k_ph_y, new_w

# проверка на выполнение кода
# c=Parametrs(c=3*(10**8),f_c=10*(10**9),Xmax=15,Ymax=15,spectr_w=1*(10**9),ph_c=(0*math.pi)/180,ang=30)
# v_cos,v_sin,v_exp,v_complx,v_floor=c.vector_func()








        