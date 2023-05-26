# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:34:34 2022

@author: Svetlana V. Morozova

"""
#open('granici_i_predeli.txt',  encoding='utf-8')

def words_from_txt(file, encoding1, wordcloud=True, stopwords=True):
    """
    words_from_txt(file, encoding1) function.
    Parameters
    ----------
    file : TYPE
        Указывается путь к текстовому файлу в формате txt.
    encoding1 : TYPE
        Указывается кдировка. При использовании кодировки utf-8 в файлах .csv и .txt: если 0-й элемент в str считывается как '\ufeff', смените кодировку с utf-8 на utf-8-sig
    stopwords : TYPE 
        False or True. Нужно ли применять список стоп-слов из библиотеки NLTK. По умолчанию True. 
        
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
        for fline in f:
            fline = re.sub('[0123456789!{|[\]-}~&(/"#—«$,»%)*+:;<=>?©^_`]','',fline) #удаляем всякую фигню из текущей строки (fline). Числа надо удалять сразу, т.к. сноскии иначе будут парсится как часть слова
            fline = fline.replace('- ','') #удаляем переносы слов, собираем слова с переносами
            myarray=fline.split() #разбиваем на слова 
            for word in myarray: #list2str
                p = morph.parse(word)[0].tag
                if 'NPRO' in p: # сохраняем  местоимения-существительные
                    mytext = mytext+' '+morph.parse(word)[0].normal_form #Бежим по словам добавляем нормальную форму слов в mywords
                elif word == 'не': # сохраняем отрицания
                    mytext = mytext+' '+morph.parse(word)[0].normal_form
                elif len(word) > 2: #для слов, состоящих из более чем 3 букв
                    mytext = mytext+' '+morph.parse(word)[0].normal_form
                else:
                    mytext = mytext


    print('морфологический анализ текста выполнен')
    
    mytext1 = mytext.replace('.','') #удаляем точки из текста
    mytext2 = mytext1.replace('ё','е') #заменяем буквы ё на буквы е (т.к. в словарях нет вариантов написания с буквами ё)
    
    mywords = mytext2.split() #вытаскиваем слова в нормальной форме
    print('токенизация выполнена')
    print()
    
    if stopwords==True: #список служебных слов вязт из библиотеки NLTK
        nltk_stopwords = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
                          'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
                          'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
                          'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
                          'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
                          'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
                          'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
                          'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
                          'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой',
                          'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая',
                          'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
                          'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']
        mywords = [word for word in mywords if word not in nltk_stopwords] #чистим слова от русских стоп-слов
    else:
        if stopwords==False:
            mywords = mywords
    
    
    if wordcloud == True:
        from wordcloud import WordCloud
        #визуализируем токены без биграммы токенов, 50 наиболее часто встречающихся, включаем слова от 3 букв.
        wordcloud = WordCloud(background_color ='white', colormap='twilight_shifted', 
                          width = 1000, height = 500, max_words=50, collocations=False, 
                          min_word_length=3).generate(" ".join(mywords))
        img = wordcloud.to_image()
        img.show()
    
    return mywords

def words_from_txt1(file, encoding1, wordcloud=True, stopwords=True):
    """
    words_from_txt(file, encoding1) function.

    Parameters
    ----------
    file : TYPE
        Указывается путь к текстовому файлу в формате txt.
    encoding1 : TYPE
        Указывается кдировка. При использовании кодировки utf-8 в файлах .csv и .txt: если 0-й элемент в str считывается как '\ufeff', смените кодировку с utf-8 на utf-8-sig
    stopwords : TYPE 
        False or True. Нужно ли применять список стоп-слов из библиотеки NLTK. По умолчанию True. 
        
    Returns
    -------
    filtered_words : list
        Возвращает список токенов: всех слов текста в словарной форме.

    """
    
    import pymorphy2
    import re
    
    morph = pymorphy2.MorphAnalyzer() #приводим слова в нормальную форму, затем собираем их в string
    
    with open(file, encoding=encoding1) as f:
        mytext='' #создаем str
        for fline in f:
            fline = re.sub('[0123456789!-{|[\]-}~&(/"#№—«$,»%)*+:;<=>?©^_`]','',fline) #удаляем всякую фигню из текущей строки (fline). Числа надо удалять сразу, т.к. сноскии иначе будут парсится как часть слова
            fline = fline.replace('- ','') #удаляем переносы слов, собираем слова с переносами
            myarray=fline.split() #разбиваем на слова 
            for word in myarray: #list2str
                p = morph.parse(word)[0].tag
                if 'NPRO' in p: # сохраняем  местоимения-существительные
                    mytext = mytext+' '+morph.parse(word)[0].normal_form #Бежим по словам добавляем нормальную форму слов в mywords
                elif word == 'не': # сохраняем отрицания
                    mytext = mytext+' '+morph.parse(word)[0].normal_form
                elif len(word) > 2: #для слов, состоящих из более чем 3 букв
                    mytext = mytext+' '+morph.parse(word)[0].normal_form
                else:
                    mytext = mytext


    print('морфологический анализ текста выполнен')
    
    mytext1 = mytext.replace('.','') #удаляем точки из текста
    mytext2 = mytext1.replace('ё','е') #заменяем буквы ё на буквы е (т.к. в словарях нет вариантов написания с буквами ё)
      
    
    import nltk
    mywords = nltk.word_tokenize(mytext2, language='russian', preserve_line=False) #вытаскиваем слова (токены)
    #preserve_line => false - НЕ сохранять номера строк в выходные данные
    print('токенизация выполнена')
    print()
    
    if stopwords==True:
        from nltk.corpus import stopwords
        mywords = [word for word in mywords if word not in stopwords.words('russian')] #чистим слова от русских стоп-слов
    else:
        if stopwords==False:
            mywords = mywords
    
    
    if wordcloud == True:
        from wordcloud import WordCloud
        #визуализируем токены без биграммы токенов, 50 наиболее часто встречающихся, включаем слова от 3 букв.
        wordcloud = WordCloud(background_color ='white', colormap='twilight_shifted', 
                          width = 1000, height = 500, max_words=50, collocations=False, 
                          min_word_length=3).generate(" ".join(mywords))
        img = wordcloud.to_image()
        img.show()
    
    return mywords


def count_dict_analysis(words:list, dict1:dict):
    """
    count_dict_analysis(words:list, dict1:dict) function.
    
    Parameters
    ----------
    words : list
        Accepts a list of tokens in the format list.
    dict1 : dict
        Accepts a dictionary of words (=keys) with categories (=values) in dict format.

    Returns
    -------
    count : dict
        Returns a dictionary with categories (=keys) and their frequency (=values) in dict format.

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
        utf-8 etc.
    delimiter2 : TYPE
        ';' or other.

    Returns
    -------
    mydict : dict
        Returns a dictionary with tokens (=keys) and their categories (=values) in dict format.

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
        utf-8 etc.
    delimiter2 : TYPE
        ';' or other.
    n : int
        n values. n<=5

    Raises
    ------
    ValueError
        n>5, n must be < or = 5.

    Returns
    -------
    mydict : dict
        User dictionary in dict with several values.

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


def dictn_part_csv(file, encoding2, delimiter2, part:int):
    """
    Loading user dictionary with one key and one value (1 from n).

    Parameters
    ----------
    file : TYPE
        File csv. The table has several columns.
    encoding2 : TYPE
        DESCRIPTION.
    delimiter2 : TYPE
        DESCRIPTION.
    part : int
        part of values (one)

    Returns
    -------
    mydict : dict
        user dictionary in dict with one value (1 from n).

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
            value = i[part]
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
#    mydict_test3 = dictn_part_csv(r'd:\mydata\words_all_full_rating.csv', encoding2='ANSI', delimiter2=';', part=3)
#    mydict_test3.pop('Words') #удаляем первую строку, в которой были названия столбцов словаря
    
#    datawords = words_from_txt('Anan2019.txt', encoding1='utf-8-sig')
#    print('токенизировали')
    
#    count_sens_words = count_dict_analysis(datawords, mydict_test3)
    
#    mydict_test3_plot = countdictplot(count_sens_words,title='Частотный анализ категорий ощущений различной модельности', lang='eng', rotat=90)

