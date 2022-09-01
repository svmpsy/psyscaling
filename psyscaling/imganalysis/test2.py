# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:40:16 2022

@author: msv-g
"""

    
def my1(a:list):
    """ Out list
    >>> my1([2,3,4])
    [2, 3, 4]
    """
    y=[]
    for x in a:
        y.append(x)
    
    return y
   
if __name__ == "__main__":
    import doctest      
    doctest.testmod(verbose=True)

