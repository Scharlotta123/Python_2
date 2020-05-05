#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Задание:
Используя данные из обучающего датасета (train.csv), построить модель для предсказания цен на недвижимость (квартиры).
С помощью полученной модели, предсказать цены для квартир из тестового датасета (test.csv).

Целевая переменная:
Price

Метрика качества:
R2 - коэффициент детерминации (sklearn.metrics.r2_score)

Требования к решению:
1. R2 > 0.6
2. Тетрадка Jupyter Notebook с кодом Вашего решения, названная по образцу {ФИО}_solution.ipynb, пример SShirkin_solution.ipynb
3. Файл CSV с прогнозами целевой переменной для тестового датасета, названный по образцу {ФИО}_predictions.csv,
пример SShirkin_predictions.csv 
Файл должен содержать два поля: Id, Price и в файле должна быть 5001 строка (шапка + 5000 предсказаний).

Сроки сдачи:
Cдать проект нужно в течение 72 часов после окончания последнего вебинара. Оценки работ, сданных до дедлайна, будут представлены в виде рейтинга, ранжированного по заданной метрике качества. Проекты, сданные после дедлайна или сданные повторно, не попадают в рейтинг, но можно будет узнать результат.

Рекомендации для файла с кодом (ipynb):
1. Файл должен содержать заголовки и комментарии (markdown)
2. Повторяющиеся операции лучше оформлять в виде функций
3. Не делать вывод большого количества строк таблиц (5-10 достаточно)
4. По возможности добавлять графики, описывающие данные (около 3-5)
5. Добавлять только лучшую модель, то есть не включать в код все варианты решения проекта
6. Скрипт проекта должен отрабатывать от начала и до конца (от загрузки данных до выгрузки предсказаний)
7. Весь проект должен быть в одном скрипте (файл ipynb).
8. Допускается применение библиотек Python и моделей машинного обучения,
которые были в данном курсе.

Описание датасета:
Id - идентификационный номер квартиры
DistrictId - идентификационный номер района
Rooms - количество комнат
Square - площадь
LifeSquare - жилая площадь
KitchenSquare - площадь кухни
Floor - этаж
HouseFloor - количество этажей в доме
HouseYear - год постройки дома
Ecology_1, Ecology_2, Ecology_3 - экологические показатели местности
Social_1, Social_2, Social_3 - социальные показатели местности
Healthcare_1, Helthcare_2 - показатели местности, связанные с охраной здоровья
Shops_1, Shops_2 - показатели, связанные с наличием магазинов, торговых центров
Price - цена квартиры


# ### Импорты

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ### Загружаем данные

# In[3]:


Train_df = pd.read_csv("train.csv")
Test_df = pd.read_csv("test.csv")


# In[4]:


Test_df.shape


# In[5]:


Train_df.shape


# ### Рассматриваем данные

# In[6]:


Train_df.info()


# In[7]:


# нуливых значений нет, тип данных разный


# In[8]:


Train_df.describe()


# In[9]:


# Мы видим только количественные признаки.
# Из их просмотра делаю следующие выводы:
#     1. В признаках LifeSquare, Healthcare_1 количество строк меньше 10000 - значит, в этих признаках есть пропущенные 
#     значения. Однако их больше половины, поэтому удалять эти столбцы не буду - заполню средними значениями.
#     2. Square - общая площадь квартиры, LifeSquare - жилая площадь. Жилая площадь должна быть меньше общей площади. 
#     3. Считаю 117 этажей в доме (мах для HouseFloor) неправдоподобными данныеми.


# In[10]:


# Рассматриваю категориальные признаки
Train_df.describe(include=[object])
# Из полученных данных видим, что эти признаки имеют 2 значения, из которых самое  
# частовстречаемое значение В. Далее убедимся, что признаки бинарные.


# In[11]:


categor_priznak = [a for a in Train_df.columns if Train_df[a].dtype.name == 'object']
for a in categor_priznak:
    print (Train_df[a].unique())


# In[12]:


Test_df.info()


# In[13]:


# Видим, что в тестовых данных также есть пропущенные данные в LifeSquare и Healthcare_1, а также видим категориальные признаки
# Ecology_2 Ecology_3 Shops_2 


# ### Заполняем пустые данные

# In[14]:


#  Пустные данные найдены в признаках LifeSquare, Healthcare_1 


# In[15]:


LifeSquare_mean = Train_df["LifeSquare"].mean()
print(LifeSquare_mean)
Train_df["LifeSquare"].fillna(LifeSquare_mean, inplace=True)


# In[16]:


Healthcare_1_mean= Train_df["Healthcare_1"].mean()
print(Healthcare_1_mean)
Train_df["Healthcare_1"].fillna(Healthcare_1_mean, inplace=True)


# In[17]:


LifeSquare_mean = Test_df["LifeSquare"].mean()
print(LifeSquare_mean)
Test_df["LifeSquare"].fillna(LifeSquare_mean, inplace=True)


# In[18]:


Healthcare_1_mean= Test_df["Healthcare_1"].mean()
print(Healthcare_1_mean)
Test_df["Healthcare_1"].fillna(Healthcare_1_mean, inplace=True)


# ### Приводим категориальные признаки к количественному типу 

# In[19]:


# Так как наши признаки Ecology_2 Ecology_3 Shops_2 бинарные, мы приводим их к значеним 0 и 1
Train_df.at[Train_df['Ecology_2'] == 'A', 'Ecology_2'] = 0
Train_df.at[Train_df['Ecology_2'] == 'B', 'Ecology_2'] = 1
Train_df['Ecology_2'].describe()


# In[20]:


Test_df.at[Test_df['Ecology_2'] == 'A', 'Ecology_2'] = 0
Test_df.at[Test_df['Ecology_2'] == 'B', 'Ecology_2'] = 1
Test_df['Ecology_2'].describe()


# In[21]:


Train_df.at[Train_df['Ecology_3'] == 'A', 'Ecology_3'] = 0
Train_df.at[Train_df['Ecology_3'] == 'B', 'Ecology_3'] = 1
Train_df['Ecology_3'].describe()


# In[22]:


Test_df.at[Test_df['Ecology_3'] == 'A', 'Ecology_3'] = 0
Test_df.at[Test_df['Ecology_3'] == 'B', 'Ecology_3'] = 1
Test_df['Ecology_3'].describe()


# In[23]:


Train_df.at[Train_df['Shops_2'] == 'A', 'Shops_2'] = 0
Train_df.at[Train_df['Shops_2'] == 'B', 'Shops_2'] = 1
Train_df['Shops_2'].describe()


# In[24]:


Test_df.at[Test_df['Shops_2'] == 'A', 'Shops_2'] = 0
Test_df.at[Test_df['Shops_2'] == 'B', 'Shops_2'] = 1
Test_df['Shops_2'].describe()


# ### Рассматриваем графические представления

# In[25]:


sns.pairplot(Train_df.select_dtypes(include='float64'))
# Данное отображение дает общее представление о распределении признаков, выбросы присутствуют


# In[26]:


plt.figure(figsize=(20,15))
sns.set(font_scale=1.4)
sns.heatmap(Train_df.corr(), annot=True, linewidth=0.5, cmap='GnBu')
plt.title('Матрица корреляции')
plt.show()


# In[68]:


# Из данных матрицы можно сделать следующие выволды:
#     - Цена напрямую зависима от количества комнат и площади,
#     - На цену довольно сильно влияет район, социальные факторы, фактор здоровья-1,
#     - На цену почти не влияет год постройки - это странно,в реальности это не так (возможно, именно дата и не влияет,
#       наверное, этот показатель был бы весомее, если бы трактовался как Возраст дома -мое предположение),
#     - На цену почти не влияет экологический фактор 1.
#     - Также мы видим меньшее влияние жилой площади, этажности, площади кухни
#     - Мы также видим зависимость социальных признаков между собой, зависимость площади от количетсва комнат, а тажке 
#     зависимость показаетелей наличия магазинов от социальных показателей


# ### Разбиваем данные

# In[ ]:


#### Разбиваем тренировочные данные на массивы х и y, где х - тренировочные признаки, оказывающие влияние на y  - цену 
#### (цена - целевое значение).
#### В таблицу признаков не включаю столбец id, так как в поясниении написано, что id - это идентификационный номер квартиры, 
#### и влияния на цену он оказывать не может
#### Из тестовых данныех также убираем столбец id


# In[27]:


x = pd.DataFrame (Train_df, columns=["DistrictId", 'Rooms', 'Square', 'LifeSquare', 'KitchenSquare',
       'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1', 'Ecology_2',
       'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 'Healthcare_1',
       'Helthcare_2', 'Shops_1', 'Shops_2'])
x.info()


# In[28]:


y = pd.DataFrame (Train_df, columns=['Price'])
y.info()


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# ### Сторим модели при обучении с учителем, выбираем лучшую

# #### Linear Regression

# In[19]:


lr = LinearRegression()
lr.fit(x_train, y_train)


# In[20]:


y_pred = lr.predict(x_test)
y_pred.shape


# In[21]:


r2_score (y_test, y_pred)


# #### значение r2 очень маленькое, что говорит о том, что модель дает не точные предстказания

# #### Random Forest

# In[30]:


rd=RandomForestRegressor()
rd.fit(x_train, y_train)


# In[31]:


y_pred=rd.predict(x_test)
r2_score (y_test, y_pred)


# #### значение r2 больше 0,6 - эта модель дает более точные предсказания, чем модель линейной регрессии

# In[163]:


# Модели knn и svc не получилось использовать - помешала ошибка Unknown label type: 'continuous', которую устранить не смогла


# In[91]:


# scaled = scaler.fit_transform(y_train)
# inversed = scaler.inverse_transform(scaled)


# In[32]:


# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# knn.fit(x_train, y_train.astype(int))
# y_pred=knn.predict(x_test)
# r2_score (y_test, y_pred)


# In[66]:


from sklearn.model_selection import GridSearchCV


# In[86]:


gb_model=RandomForestRegressor(random_state=21)
parameters = {
    'n_estimators': [150, 200, 250],
    'max_features': np.arange(5, 9),
    'max_depth': np.arange(5, 10),
}

clf = GridSearchCV(
    gb_model,
    parameters,
    scoring='r2',
    cv=10
)


# In[87]:


clf.fit(x_train, y_train)


# In[88]:


clf.best_params_


# In[89]:


clf = RandomForestRegressor(max_depth=9, max_features=8, n_estimators=200)
clf.fit(x_train, y_train)


# In[101]:


y_pred=clf.predict(x_test)
r2_score (y_test, y_pred)


# In[ ]:


### результат не тот, что ожидала- r2 стал еще меньше


# #### Запускаем модель на тестовых данных Test_df

# In[33]:


test = pd.DataFrame (Test_df, columns=["DistrictId", 'Rooms', 'Square', 'LifeSquare', 'KitchenSquare',
       'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1', 'Ecology_2',
       'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 'Healthcare_1',
       'Helthcare_2', 'Shops_1', 'Shops_2']) 
### убрала id из тестовых данных


# In[34]:


y_result=rd.predict(test)
result=pd.DataFrame (y_result, columns=['Prise'])


# In[35]:


result.info()


# In[36]:


result.to_csv('SharikovaOA_predictions.csv')


# In[ ]:




