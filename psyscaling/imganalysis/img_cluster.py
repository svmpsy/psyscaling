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


def Luscher_clasters(img:str, part:int):
    """
    Оценивает расстояние от цветов Люшера (центроидов) до каждого пикселя изображения.
    Предварительно изображение преобразуется в квадратное с определенным количеством пикселей.
    Процедура сжатия производится усреднением цветов по каналам R, G, B.
    Расстояние между центроидами и пикселями расчитываются с помощью квадрата евклидова расстояния.

    Parameters
    ----------
    img : str
        Адрес изображения.
    part : int
        Изображение преобразуется в квадратное. 
        Параметр указывает количество пикселей по каждой стороне. 
        Не может быть больше исходного размера изображения.

    Returns
    -------
    aovdt : pd.DataFrame
        Датафрейм, включающий дистанции от цветов Люшера и номера центроидов.

    """
    
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

    Luscher_centroid = np.array([[30., 40., 100.],
                                  [35., 105., 115.],
                                  [240., 80., 55.],
                                  [255., 250., 105.],
                                  [175., 55., 105.],
                                  [155., 115., 95.],
                                  [0., 0., 0.],
                                  [185., 185., 185.]])

                   
    code,distance = vq(features,Luscher_centroid) #вычисляем евклидово расстояние между пикселями изображения и центроидами
    codeim = code.reshape(steps,steps)

    plt.figure()
    plt.imshow(codeim)
    plt.show()
    #plt.savefig('c:/mydata/image/kmeanspic.png') #сохраняем диаграмму в файл

    #круговая диаграмма, cnts-размер в пикселях
    a, cnts = np.unique(codeim, return_counts=True)
    percent = np.array(cnts)*100/np.sum(cnts)
    plt.pie(percent,colors=np.array(Luscher_centroid/255),labels=a)
    plt.show()
    perc = pd.DataFrame({'claster':a,'percent':percent})
    print('проценты встречаемости цветов Люшера в изображении')
    print(perc)
    print()

    cmax = max(percent)
    ind_max_centroid = np.where(percent == cmax)
    d1=ind_max_centroid[0]
    d2=Luscher_centroid[d1[0]]
    Luscher_RGB = pd.DataFrame({'claster':d1,'R':d2[0],'G':d2[1],'B':d2[2]})
    print('самый часто встречающийся цвет:')
    print(Luscher_RGB)
    print()
    
    aovdt = pd.DataFrame({'distance':distance,'code':code})
    
    return aovdt


def aov_claster(df:pd.DataFrame):
    """
    Оценивает различия центроидов по степени отклонения от них пикселей.
    Вычисляет однофакторный дисперсионный анализ, апостериорные сравнения,
    критерий Левина. Выводит графики: boxplot c mean,QQ-plot.

    Parameters
    ----------
    df : pd.DataFrame
        Результаты выполнения функции Luscher_clasters().

    Returns
    -------
    None.

    """
    
    from bioinfokit.analys import stat
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import seaborn as sns

    aovdt = df
    
    
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

    sm.qqplot(res.anova_std_residuals, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Standardized Residuals")
    plt.show()

    res = stat()
    res.levene(df=aovdt, res_var='distance', xfac_var='code')
    print('Levene’s test')
    print(res.levene_summary)
    print()

    res = stat()
    res.tukey_hsd(df=aovdt, res_var='distance', xfac_var='code', anova_model='distance ~ C(code)')
    print('POST-HOC')
    print(res.tukey_summary)
    print()



def kwtest(df:pd.DataFrame):
    """
    Выводит графики: barplot c median.
    Также выводятся результаты непараметрического критерия H Kruskal–Wallis.

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    from scipy import stats
    import seaborn as sns
    from numpy import median
    
    aovdt = df
    
    kw=stats.kruskal(aovdt['code'] == 0, aovdt['code'] == 1, aovdt['code'] == 2, 
                  aovdt['code'] == 3, aovdt['code'] == 4, aovdt['code'] == 5,
                  aovdt['code'] == 6, aovdt['code'] == 7)
    print('Kruskal–Wallis')
    print(kw) 
    print()
    
    plt2=sns.barplot(x='code', y='distance', data=aovdt, color='gray', estimator = median)
    plt2

    
def brightness_hist(img:str,color='L',title='Яркость изображения',
                    xlb='Яркость',ylb='Количество пикселей'):
    """
    Гистограмма яркости изображения. Палитры: grey or RGB.
    Оцениваем среднюю яркость изображения.

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

#    distance_Luscher_clasters=Luscher_clasters(r'd:\img\test1.jpg',500)
#    aov_claster(distance_Luscher_clasters)
#    kwtest(distance_Luscher_clasters)
#    mean_color=brightness_hist(r'd:\img\test1.jpg',color='RGB')
    
