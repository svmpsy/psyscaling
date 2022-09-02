# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:34:34 2022

@author: Svetlana V. Morozova

"""
#open('granici_i_predeli.txt',  encoding='utf-8')

def words_from_txt(file, encoding1):
    """
    words_from_txt(file, encoding1) function.

    Parameters
    ----------
    file : TYPE
        Указывается путь к текстовому файлу в формате txt.
    encoding1 : TYPE
        Указывается кдировка. При использовании кодировки utf-8 в файлах .csv и .txt: если 0-й элемент в str считывается как '\ufeff', смените кодировку с utf-8 на utf-8-sig

    Returns
    -------
    filtered_words : list
        Возвращает список токенов: всех слов текста в словарной форме.

    """
    
    import pymorphy2
    import re
    
    morph = pymorphy2.MorphAnalyzer() #приводим слова в нормальную форму, затем собираем их в string
    
    with open(file, encoding=encoding1) as f:
        mytext='' #создаем list
        skipstr = 0 #ввожу маркер пустой строки (см. строки 19,20 и затем 16,17)
        for fline in f:
            if skipstr == 1: #если была пустая строка
                skipstr = 0 #пропускаем ввод слов в mytext, чтобы избавиться от верхних колонтитулов. Возможно, можно убрать постраничные ссылки, закодив поиск списка строк, начинающихся с цифр, перед пустой строкой (концом страницы)
            else:
                if len(fline) == 0: #если текущая строка пустая
                    skipstr = 1 #ставим маркер, что следующую строку пропустить и дальше читаем текущую строку
                fline = re.sub('[0123456789!{|[\]-}~&(/"#—«$,»%)*+:;<=>?©^_`]','',fline) #удаляем всякую фигню из текущей строки (fline). Числа надо удалять сразу, т.к. сноскии иначе будут парсится как часть слова
                fline = fline.replace('- ','') #удаляем переносы слов, собираем слова с переносами
                myarray=fline.split() #разбиваем на слова 
                for word in myarray: #list2str
                    if len(word)>3: #для слов, состоящих из более чем 3 букв
                        mytext = mytext+' '+morph.parse(word)[0].normal_form #Бежим по словам добавляем нормальную форму слов в mywords
    #print('создали mytext')

    mytext1 = mytext.replace('.','') #удаляем точки из текста
      
    
    import nltk
    mywords = nltk.word_tokenize(mytext1, language='russian', preserve_line=False) #вытаскиваем слова (токены)
    #preserve_line => false - НЕ сохранять номера строк в выходные данные
      
    from nltk.corpus import stopwords
    filtered_words = [word for word in mywords if word not in stopwords.words('russian')] #чистим слова от русских стоп-слов
    
    return filtered_words




def count_dict_analysis(words:list, dict1:dict):
    """
    count_dict_analysis(words:list, dict1:dict) function.
    
    Parameters
    ----------
    words : list
        Принимает список токенов в формате list.
    dict1 : dict
        Принимает словарь слов (=keys) с категориями (=values) в формате dict.

    Returns
    -------
    count : dict
        Возвращает словарь с категориями (=keys) и частотой их встречаемости (=values) в формате dict.

    """
    
    count = {}
    count1={}
    v=0
    for i in words:
        for key, value in dict1.items():
            if key == i:
                if value in count:
                    v=count[value]+1
                else:
                    v=1
                count1[value]=v
                count.update(count1)
    #            print(count) #проверяем как бежит, выводя обновления словаря на каждой итерации
            count1={}
    return count


def dict2_csv(file, encoding2, delimiter2):
    """
    Loading user dictionary with one key and one value.

    Parameters
    ----------
    file : TYPE
        File csv. The table has only two columns!
    encoding2 : TYPE
        DESCRIPTION.
    delimiter2 : TYPE
        DESCRIPTION.

    Returns
    -------
    mydict : dict
        DESCRIPTION.

    """
    import csv
    mydict2 = {}
    with open(file, encoding=encoding2, newline='') as f:
        reader = csv.reader(f, delimiter=delimiter2)
        mydict2 = dict(reader)
    return mydict2


def dictn_csv(file, encoding2, delimiter2, n:int):
    """
    Loading user dictionary with one key and 2 to 5 values.

    Parameters
    ----------
    file : TYPE
        File csv. The table has several columns.
    encoding2 : TYPE
        DESCRIPTION.
    delimiter2 : TYPE
        DESCRIPTION.
    n : int
        n values. n<=5

    Raises
    ------
    ValueError
        n>5, n must be < or = 5.

    Returns
    -------
    mydict : dict
        user dictionary in dict with several values.

    """

    import csv
        
    mydict1 = []
    mydict = {}
    mydict2 = {}
    with open(file, encoding=encoding2) as f:
        reader = csv.reader(f, delimiter=delimiter2)
        mydict1 = list(reader)
        for i in mydict1:
            value = []
            key = i[0]
            if n==2:
                value = i[1], i[2]
            elif n==3:
                value = i[1], i[2], i[3]
            elif n==4:
                value = i[1], i[2], i[3], i[4]
            elif n==5:
                value = i[1], i[2], i[3], i[4], i[5]
            else:
                raise ValueError('n>5, n must be < or = 5')
            mydict2[key]=value
            mydict.update(mydict2)
        mydict2={}
    return mydict


def countdictplot(dictionary:dict, lang:str, title:str, rotat:int):
    """
    Visualization of frequencies for the analyzed text on a plot.

    Parameters
    ----------
    dictionary : dict
        Frequency dictionary for the analyzed text.
    lang : str
        'rus' or 'eng'
    title : str
        Title of plot.
    rotat : int
        The angle of the x-axis labels. Integer.

    Returns
    -------
    None.

    """
    
    import matplotlib.pyplot as plt
    
    if lang=='rus':
        plt.bar(range(len(dictionary)), list(dictionary.values()), align='center')
        plt.xticks(range(len(dictionary)), list(dictionary.keys()))
        yint = []
        locs, labels = plt.yticks()
        for each in locs:
            yint.append(int(each))
        plt.yticks(yint)
        plt.ylabel("Частота") #добавляем названия осей и диаграммы
        plt.xticks(rotation=rotat)
        plt.xlabel("Категории")
        plt.title(title) # Заголовок диаграммы
    else:
        plt.bar(range(len(dictionary)), list(dictionary.values()), align='center')
        plt.xticks(range(len(dictionary)), list(dictionary.keys()))
        yint = []
        locs, labels = plt.yticks()
        for each in locs:
            yint.append(int(each))
        plt.yticks(yint)
        plt.ylabel("Frequency") #добавляем названия осей и диаграммы
        plt.xticks(rotation=rotat)
        plt.xlabel("Categories")
        plt.title(title) # Заголовок диаграммы



#if __name__ == "__main__":

#    mydict_test1 = dict2_csv(r'd:\mydata\sensdict_young.csv', encoding2='utf-8-sig', delimiter2=';')
#    mydict_test2 = dictn_csv(r'd:\mydata\rusentilex_2017.csv', encoding2='utf-8-sig', delimiter2=';', n=4)
    
#    datawords = words_from_txt('Anan2019.txt', encoding1='utf-8-sig')
#    print('токенизировали')
    
#    count_sens_words = count_dict_analysis(datawords, mydict_test2)
    
#    mydict_test2 = countdictplot(count_sens_words,title='Частотный анализ категорий ощущений различной модельности', lang='eng', rotat=90)

