import numpy as np
import pandas as pd

from math import trunc

from numpy import nan as NA
from pandas import  Series
from pandas import  DataFrame

#import cx_Oracle 
#import os
#os.putenv('NLS_LANG','KOREAN_KOREA.KO16MSWIN949') 

from datetime import datetime
from datetime import timedelta
import pandas.tseries.offsets


import re
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Day, Hour, Second

#0414
def f_write_txt(vname, fname, sep=' ' , fmt= '%.2f'):
    c1 = open(fname,'w')
    for i in vname:
        vstr=''
        for j in i:
            j = fmt % j
            vstr = vstr + j + sep
        vstr = vstr.rstrip(sep)    
        c1.writelines(vstr + '\n')
    c1.close()
    

import mglearn

    
# 데이터 셋
from sklearn.datasets import load_iris as iris
from sklearn.datasets import load_breast_cancer as cancer

# 분석
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import KNeighborsRegressor as knn_R

from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.tree import DecisionTreeRegressor as dt_r

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import RandomForestRegressor as rf

from sklearn.ensemble import GradientBoostingClassifier as gb
from sklearn.ensemble import GradientBoostingRegressor as gb_r



# feature별 영향도 시각화
def plot_feature_importances(model, data):
    n_features = data.data.shape[1]
    plt.barh(range(n.features), mode.feature_importances_, align='center')
    ply.yticks(np.arange(n_features), data.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)


#[시각화]
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'

from sklearn.tree import export_graphviz
import graphviz          # 설치 필요

# 교호작용
from sklearn.preprocessing import PolynomialFeatures

# 스케일 조정
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# SVM
from sklearn.svm import SVC
# PCA
from sklearn.decomposition import PCA
# 얼굴인식 데이터셋
from sklearn.datasets import fetch_lfw_people

#CV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold