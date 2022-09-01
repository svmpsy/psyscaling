# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 00:30:20 2022

@author: msv-g
"""

def myfunction4(a:list):
    """
    >>> myfunction4([3, 2, 1])
    True
    """
    flag = False
    for x in a:
        if x == 1:
            flag = True
    return flag # функция возвращает флаг

if __name__ == "__main__":
    import doctest      
    doctest.testmod(verbose=True)