<!---
title: "README"
date: '2024-09-30'
author: "Morozova S.V."
--->

**PACKAGE "PSYSCALING" FOR PYTHON**

**- Info**

*Psychological scaling of text and graphic information. Used to prepare experimental data for quantitative analysis. Supports only Russian language now.*

---
**- setup**

*pip install git+https://github.com/svmpsy/psyscaling.git*


If there are problems, you can first install git:
*pip install git или conda install -c anaconda git*


Простая установка библиотеки psyscaling из архива (через conda prompt, **актуальна в случае, если установщик git выдает ошибку**):

*pip install https://github.com/svmpsy/psyscaling/archive/refs/heads/main.zip*


**You can also download the archive from GitHub and install it offline.**

*Some functions require libraries: nltk, wordcloud, re, csv, cv2(opencv-python), scipy, seaborn, bioinfokit, statsmodels, pingouin.*

---
**- discr**

При использовании кодировки utf-8 в файлах .csv и .txt: если 0-й элемент в str считывается как '\ufeff', смените кодировку с utf-8 на utf-8-sig

---
**- сiting**

Морозова С.В. Анализ текстов и изображений в психологических исследованиях с помощью библиотеки “psyscaling” // Ананьевские чтения – 2023. 60 лет социальной психологии в СПбГУ: Человек в современном мире: потенциалы и перспективы психологии развития. - Б.м.: «Кириллица», 2023. - С. 140

---
**- copyright**

*Copyright (C) 2022-2024, S.V. Morozova*
