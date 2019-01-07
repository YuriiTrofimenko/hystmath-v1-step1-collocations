# пакет для символьной и статистической обработки естественного языка
import nltk
# перед первым запуском раскомментировать, чтобы загрузились готовые наборы стоп-слов
# (которые нужно исключить из исследуемой выборки)
# nltk.download('stopwords')

"""
Перед вторым запуском раскомментировать,
чтобы считать в переменную тексты из файлов-источников
"""

"""
# пакет для работы с файловой системой
import os

# пустая строка для накопления исходных текстов
s = ''
# относительный путь к каталогу с исходными текстами
Path = "./hm_texts/raw/semenov/2_contradictions/"
# получение списка файлов с исходными текстами
filelist = os.listdir(Path)
# чтение слов из всех файлов с текстовым расширением в одну переменную
for i in filelist:
    if i.endswith(".txt"):
        with open(file=(Path + i), mode='r', encoding="utf8") as f:
            for line in f:
                s += line.strip() + ' '
# получение наборов стандартных стоп-слов
from nltk.corpus import stopwords
# объединение готовых наборов стоп-слов
mystopwords = stopwords.words('russian') + stopwords.words('english') + stopwords.words('german')
# пакет работы с регулярными выражениями
import re
# регулярное выражение, разрешающее только буквенные символы русского алфавита
prog = re.compile('[А-Яа-я]+')
# приведение всех символов к нижнему регистру и фильтрация регулярным выражением
l1 = prog.findall(s.lower())

# пакет для исследования текстов
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
# приведение слов к нормальной форме
l3 = [morph.parse(token)[0].normal_form for token in l1 if not morph.parse(token)[0].normal_form in mystopwords]
"""

# сохранение промежуточных текстовых материалов в файл (optional)
"""f3 = open(file='semenov_contradictions_intermediate.txt', mode='w', encoding="utf8")
for item in l3:
    f3.write("%s\n" % item)
f3.close()"""

"""
# определение наиболее часто встречающихся слов из трех букв
short3 = []
for t in l3:
    if len(t) > 2 and len(t) < 4 :
        short3.append(t)
dd = nltk.FreqDist(short3)
words_most_frequent = dd.most_common(20)
#  words_most_frequent = dd.keys()
print(words_most_frequent)
"""

"""
# фильтрация списка слов при помощи собственного набора трехбуквенных сочетаний,
# выявленных выше
m = []
for t in l3:
    if len(t) > 2 or t in ['рф']:
        if t not in ['год','это','мир','наш','сша','ещё','век','имя','ряд','оно','тип', 'род','ввп','вид’,’ном','ибо','сми','буш','фон’,’шар']:
            m.append(t)

# сохранение дополнительно отфильтрованных промежуточных текстовых материалов в файл
filem = open('semenov_contradictions_intermediate_2.txt', 'w', encoding="utf8")
for item in m:
    filem.write("%s\n" % item)
filem.close()
"""

"""
Перед заключительным запуском раскомментировать,
чтобы определить коллокации и вывести в консоль
"""


# пакет для высокоуровневой обработки и анализа данных
import pandas as pd
# настройка максимальной ширины консольного вывода
desired_width = 480
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 5)
#если используется непосредственно пакет нампи, то нужно добавить настройку ширины вывода для него:
#import numpy as np
#np.set_printoptions(linewidth=desired_width)

# массив для промежуточного набора слов для исследования
m = []
# чтение слов из файла в массив
with open('./semenov_contradictions_intermediate_2.txt', encoding="utf8") as infile:
    for line in infile:
        m.append(line[:-1])

# подключение функционала поиска коллокаций
from nltk.collocations import *
# число извлекаемых биграм
N_best = 100

# класс для мер ассоциации биграм
bigram_measures = nltk.collocations.BigramAssocMeasures()
# класс для хранения и извлечения биграм
finder = BigramCollocationFinder.from_words(m)
# избавимся от биграм, которые встречаются реже n раз
finder.apply_freq_filter(20)
# выбираем топ-100 биграм по частоте
raw_freq_ranking = [' '.join(i) for i in finder.nbest(bigram_measures.raw_freq, N_best)]
# выбираем топ-100 биграм по каждой мере
tscore_ranking = [' '.join(i) for i in finder.nbest(bigram_measures.student_t, N_best)]
pmi_ranking = [' '.join(i) for i in finder.nbest(bigram_measures.pmi, N_best)]
llr_ranking = [' '. join(i) for i in finder.nbest(bigram_measures.likelihood_ratio, N_best)]
chi2_ranking = [' '.join(i) for i in finder.nbest(bigram_measures.chi_sq, N_best)]
# образование набора результатов из пяти колонок (по одной колонке на каждый способ поиска коллокаций)
rankings = pd.DataFrame({
                            'chi2': chi2_ranking
                            , 'llr': llr_ranking
                            , 't-score': tscore_ranking
                            , 'pmi': pmi_ranking
                            , 'raw_freq': raw_freq_ranking})
rankings = rankings[['raw_freq', 'pmi', 't-score', 'chi2', 'llr']]
# отбор не более двадцати наилучших результатов для каждого из способов поиска
rankings.head(20)
# вывод колонок коллокаций в консоль
print(rankings)