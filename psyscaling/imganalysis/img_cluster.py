# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 15:32:44 2022

@author: Svetlana V. Morozova
"""

import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.vq import kmeans, vq 
import matplotlib.pyplot as plt
from bioinfokit.analys import stat

rgb_basic_colors = np.array([[0., 0., 255.], [0., 255., 0.], [255., 0., 0.],
                             [0., 255., 255.], [255., 0., 255.], [255., 255., 0.]])

Luscher_centroid = np.array([[30., 40., 100.],
                              [35., 105., 115.],
                              [240., 80., 55.],
                              [255., 250., 105.],
                              [175., 55., 105.],
                              [155., 115., 95.],
                              [0., 0., 0.],
                              [185., 185., 185.]])


def color_clasters(img:str,part:int,thumbnail=False,centroids=rgb_basic_colors):
    """
    Оценивает расстояние от заданных цветов (центроидов) до каждого пикселя изображения.
    Базовая палитра - 6 основных цветов RGB.
    Предварительно изображение преобразуется c учетом количества пикселей по большей стороне 
    (соотношение сторон сохраняется) или в квадратное.
    Расстояния между центроидами и пикселями расчитываются с помощью квадрата евклидова расстояния.

    Parameters
    ----------
    img : str
        Адрес изображения. Работает с изображениями форматов: bmp, jpg, png, tif.
    part : int
        Параметр указывает количество пикселей по большей стороне. 
        Не может быть больше исходного размера изображения.
    thumbnail : logical, optional
        Если False - изображение преобразуется в квадратное через усреднение цветности пикселей.   
        Если True, то изображение преобразуется в миниатюру с сохраненинем пропорций
        с помощью функции Image.thumbnail(). 
        Параметр part указывает количество пикселей по большей стороне. 
        Если изображение квадратное, то part указывает количество пикселей по обеим сторонам.
        The default is False.
    centroids : np.array
        Цвета центроидов. По умолчанию 6 базовых цветов RGB.

    Returns
    -------
    aovdt : pd.DataFrame
        Датафрейм, включающий дистанции от цветов палитры и номера центроидов.
        
    """
       
    if thumbnail == False:
        
        im = np.array(Image.open(img))
        print()
        print('Информация об изображении [height, width, channels]:', im.shape )
        print()
        
        steps = part
        dx = im.shape[0] / steps
        dy = im.shape[1] / steps
        dx = int(dx)
        dy = int(dy)

        features = []

        for x in range(steps): #расчитываем средние значения по каналам для каждого из n квадратов (steps)
            for y in range(steps):
                R = np.mean(im[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 0])
                G = np.mean(im[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 1])
                B = np.mean(im[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 2])
                features.append([R,G,B])
        features = np.array(features, "f")
        
        im_size = (steps,steps)
        
    if thumbnail == True:
        
        im = Image.open(img)
        print()
        print('Информация об изображении:')
        print("Size : ",im.size)
        print()
        
        n = part
        #w = float()
        #h = float()
        if im.height > im.width:
            h = n
            #w = (n*im.height)/im.width
            w = (n*im.width)/im.height
            w = int(w)
            size = (w, h)
        elif im.height < im.width:          
            w = n
            #h = (n*im.width)/im.height
            h = (n*im.height)/im.width
            h = int(h)
            size = (w, h)
        elif im.height == im.width:
            size = (n, n)

        im.thumbnail(size)
        
        im_size = (im.size[1],im.size[0])
        
        im = np.array(im)
        
        features = []

        for x in range(len(im)): #по всем столбцам массива
            for y in range(len(im[0])): #идти по всем строкам первого столбца
                R = (im[x, y, 0])
                G = (im[x, y, 1])
                B = (im[x, y, 2])
                features.append([R,G,B])
        features = np.array(features, "f")  
                   
    code,distance = vq(features,centroids) #вычисляем евклидово расстояние между пикселями изображения и центроидами
    codeim = code.reshape(im_size)

    plt.figure()
    plt.imshow(codeim)
    plt.show()
    
    #круговая диаграмма, cnts-размер в пикселях
    a, cnts = np.unique(codeim, return_counts=True)
    percent = np.array(cnts)*100/np.sum(cnts)
    
    #выбираем в палитре только существующие цвета
    l=0
    cent1=np.zeros([len(a),3])
    for i in a:
        cent1[l]=centroids[i]/255
        l=l+1
    
    #рисуем
    plt.figure()
    plt.pie(percent,colors=cent1,labels=a)
    plt.show()
    perc = pd.DataFrame({'claster':a,'percent':percent})
    print('Проценты встречаемости цветов палитры в изображении:')
    print(perc)
    print()

    cmax = max(percent)
    ind_max_centroid = np.where(percent == cmax)
    d1=ind_max_centroid[0]
    d2=centroids[d1[0]]
    Luscher_RGB = pd.DataFrame({'claster':d1,'R':d2[0],'G':d2[1],'B':d2[2]})
    print('Самый часто встречающийся цвет палитры:')
    print(Luscher_RGB)
    print()
    
    aovdt = pd.DataFrame({'distance':distance,'code':code, 'R':features[:,0], 'G':features[:,1], 'B':features[:,2]})
    
    print('---------------')
    res_claster = stat()
    res_claster.anova_stat(df=aovdt, res_var='R', anova_model='R~C(code)')
    print('Проверка статистической значимости различий кластеров по каналу R (ANOVA)')
    print(res_claster.anova_summary)
    print()
    
    res_claster = stat()
    res_claster.anova_stat(df=aovdt, res_var='G', anova_model='G~C(code)')
    print('Проверка статистической значимости различий кластеров по каналу G (ANOVA)')
    print(res_claster.anova_summary)
    print()
    
    res_claster = stat()
    res_claster.anova_stat(df=aovdt, res_var='B', anova_model='B~C(code)')
    print('Проверка статистической значимости различий кластеров по каналу B (ANOVA)')
    print(res_claster.anova_summary)
    print('---------------')
    print()
    print()
    
    return aovdt


def aov_claster(df:pd.DataFrame):
    """
    Оценивает различия центроидов по степени отклонения от них пикселей.
    Вычисляет однофакторный дисперсионный анализ, апостериорные сравнения,
    критерий Левина. Выводит графики: boxplot c mean,QQ-plot.

    Parameters
    ----------
    df : pd.DataFrame
        Результаты выполнения функции color_clasters().

    Returns
    -------
    Моду и медиану для значений попиксельных расстояний до центроидов.

    """
    
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import seaborn as sns
    from numpy import mean

    aovdt = df
    
    plt.figure()
    plt1 = sns.boxplot(x='code', y='distance', data=aovdt, color='gray')
    plt1

    res = stat()
    res.anova_stat(df=aovdt, res_var='distance', anova_model='distance~C(code)')
    aov_table=res.anova_summary
    esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
    aov_table['EtaSq'] = [esq_sm, 'NaN']
    print('ANOVA results')
    print(aov_table)
    print()
    
    plt.figure()
    sm.qqplot(res.anova_std_residuals, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    plt.show()

    res = stat()
    res.levene(df=aovdt, res_var='distance', xfac_var='code')
    print('Levene’s test')
    print(res.levene_summary)
    print()

    #res = stat()
    #res.tukey_hsd(df=aovdt, res_var='distance', xfac_var='code', anova_model='distance ~ C(code)')
    #print('POST-HOC')
    #print(res.tukey_summary)
    #print()
    tukey = pairwise_tukeyhsd(endog=aovdt['distance'], groups=aovdt['code'], alpha=0.05)
    print(tukey)
    print()
    
    descript_stat = aovdt.groupby(['code']).aggregate(['mean','median'])
    
    return descript_stat



def kwtest(df:pd.DataFrame):
    """
    Вычисляется непараметрический критерий H Kruskal–Wallis. 
    Выводит графики: barplot c median.

    Parameters
    ----------
    df : pd.DataFrame
        Результаты выполнения функции color_clasters().

    Returns
    -------
    Моду и медиану для значений попиксельных расстояний до центроидов.

    """
    
    #from scipy import stats
    import seaborn as sns
    from numpy import median
    from pingouin import kruskal
    
    aovdt = df
    
    descript_stat = aovdt.groupby(['code']).aggregate(['mean','median'])
    
    kw = kruskal(data=aovdt, dv='distance', between='code')
    
    print('Kruskal–Wallis')
    print()
    print(kw) 
    print()
    
    plt.figure()
    plt2=sns.barplot(x='code', y='distance', data=aovdt, color='gray', estimator = median)
    plt2
    
    return descript_stat

    
def brightness_hist(img:str,color='L',title='Яркость изображения',
                    xlb='Яркость',ylb='Количество пикселей'):
    """
    Гистограмма яркости изображения. Палитры: grey or RGB

    Parameters
    ----------
    img : str
        адрес изображения.
    color : TYPE, optional
        Палитра: градации серого ('L') или RGB ('RGB'). The default is 'L'.
    title : TYPE, optional
        название графика. The default is 'Яркость изображения'.
    xlb : TYPE, optional
        название оси яркости. The default is 'Яркость'.
    ylb : TYPE, optional
        название оси количества пикселей. The default is 'Количество пикселей'.

    Returns
    -------
    Среднюю яркость изображения: mean_color.

    """
    
    import imhist
    import matplotlib.pyplot as plt
    from PIL import Image
    import IPython
    import cv2
    
    im = np.array(Image.open(img).convert(color))
    IPython.display.display(Image.fromarray(im))
    
    print()
    print('Информация об изображении [height, width, channels]:', im.shape )
    print()
    
    plt.figure()
    plt.hist(imhist.imhist(im))
    plt.title('Яркость изображения')
    plt.ylabel('Количество пикселей')
    plt.xlabel('Яркость')
    plt.show()
    
    if color=='RGB':
        r,g,b = cv2.split(im)
        mean_color = [np.mean(r),np.mean(g),np.mean(b)]
    if color=='L':
        grey = cv2.split(im)
        mean_color = [np.mean(grey)]
    print('Средняя яркость изображения:', color)
    print(mean_color)
    
    return mean_color
    

#if __name__ == "__main__":  

#    distance_color_clasters=color_clasters(r'd:\img\test2.jpg',300,thumbnail=True)
#    descript_stat=aov_claster(distance_color_clasters)
#    descript_stat=kwtest(distance_color_clasters)
#    mean_color=brightness_hist(r'd:\img\Shishkin\1889.jpg',color='RGB')
