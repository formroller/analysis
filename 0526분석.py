run profile1

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


# =============================================================================
# # decision Tree - iris data set
# =============================================================================

#1. 데이터 로딩
df_iris = iris()  # iris는 load_iris 함수의 alias 처리된 것 

#2. 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(df_iris['data'],
                                                    df_iris['target'],
                                                    random_state=0)

#3. 모델 생성 및 train data 학습
m_dt = dt()
m_dt.fit(x_train, y_train)

#4. 모델 평가(test data set 적용)
m_dt.score(x_test, y_test)    # 97

#5. 매개변수 튜닝 
max_depth            : 설명변수의 재사용 횟수
                     => 값이 클수록 모델이 복잡해질 확률 증가   

max_feature          : 각 노드에서 설명변수 선택 시 고려되는 후보의 개수
                     => 값이 작을수록 서로 다른 트리 구성될 확률이 높다
                     
min_sample_split     : 최소 가지치기 개수 (오분류 건수가 해당 값 이상일 경우 추가 split)
                     => 값이 작을수록 분할이 늘어난다. - 복잡도 증가(오분류값이 min_split보다 클 경우 분기)   

랜덤포레스트
서로다른 트리 구성하기 위함
-> 학습된 데이터를 다시 랜덤하게 트리별로 학습시키기 위해 노력한다.(복원 추출허용)
-> 복원 추출 허용하며 서로다른 데이터셋에 학습
-> 중요한 질문을 앞에 던진다 ->
랜덤하게 설정된 변수중 (max_feature)
대표본 데이터일 경우 선택될           


#6. new data 예측
new_data = np.array([[5.0,2.9,3.0,1.5]])
new_data_y = m_dt.predict(new_data)[0]
df_iris['target_names'][new_data_y]

#7. 모델 기타 정보 (특성 중요도)
m_dt.feature_importances_  # gini계수 

# =============================================================================
# # [decision tree - cancer data set]
# # min_split 계수 변화에 따른 score 점수 확인
# -> 튜닝 : overfit 잡기 위해 사용
# =============================================================================

#1. 데이터 로딩
from sklearn.datasets import load_breast_cancer as cancer
dt_cancer = cancer()

#2. train, test 분리
x_train, x_test, y_train, y_test = train_test_split(dt_cancer['data'],  # 설명변수 데이터 셋
                                                    dt_cancer['target'],  # 종속변수 데이터 셋
                                                    random_state=0)   # seed값 고정
#3. 모델 생성 및 train data 학습
m_dt2=dt()
m_dt2.fit(x_train,y_train)

#4. 모델 평가 (test data set 적용)
m_dt2.score(x_test, y_test)    # 89.9

#5. 매개변수 튜닝(score)
# - min_slpit 작을수록 과대적합 심해질 수 있다 -> 새로운 데이터 예측력 떨어질 수 있다.
score_train=[]; score_test=[]
for i in np.arange(2,21):
    m_dt2 = dt(min_samples_split=i)
    m_dt2.fit(x_train, y_train)
    score_train.append(m_dt2.score(x_train,y_train))
    score_test.append(m_dt2.score(x_test, y_test))

score_train
score_test

import matplotlib.pyplot as plt

plt.plot(np.arange(2,21), score_train, label='train_scroe')
plt.plot(np.arange(2,21), score_test, label='test_scroe', color = 'red')
plt.legend()

#6. 모델 고정
m_dt2=dt(min_samples_split=12)
m_dt2.fit(x_train, y_train)
m_dt2.feature_importances_

s1 = Series(m_dt2.feature_importances_, index = df_cancer.feature_names)
s1.sort_values(ascending=False)

# =============================================================================
# # [참고 - Decision Tree 시각화 ]
# =============================================================================
# 1. window graphviz 설치
#  - https://graphviz.gitlab.io/_pages/Download/Download_windows.html 
#  - 다운로드 후 C:/Program Files (x86) 위치에 압축해제(추가 설치 필요 없음)
  
# 2. PATH 설정
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)'
# 3. anaconda package 설치
pip install graphviz

# 4. 시각화 작업
df_cancer['target_names']
from sklearn.tree import export_graphviz

export_graphviz(m_dt2,                   # 모델명 
                out_file="tree.dot", 
                class_names=df_cancer.target_names, # target name 
                feature_names=dt_cancer.feature_names, 
                impurity=False, 
                filled=True)

import graphviz          # pip install 설치 필요

with open("tree.dot", encoding='UTF8') as f:
    dot_graph = f.read()
    
g1 = graphviz.Source(dot_graph)
g1.render('a2', cleanup=True) 

# =============================================================================
# # Tree 구조 모델
# =============================================================================
  Decision Tree > Random Forest 
  > Gradiant Boosting Tree(GB) > extreme Gradiant Boosting Tree(XGB)


# =============================================================================
# Random Forest - cancer data set
# =============================================================================
#1. 모델 생성 및 학습
m_rf = rf()
m_rf.fit(x_train, y_train)   # n_estimators=10, max_feature='auto',
                             # min_samples_split=2
* n_estimators       : 트리 갯수
* max_feature='auto' : 자동 튜닝, 트리의 random을 생성하는 변수(클수록 과대적합 해소)

#2. 모델 평가
m_rf.score(x_test, y_test)  # 85.7

#3. 매개변수 튜닝
# 3-1) 적절한 tree 갯수 선택 (n_estimate)
score_train=[]; score_test=[]
for i in np.arange(1,101):
    m_rf=rf(n_estimators=i)
    m_rf.fit(x_train, y_train)
    score_train.append(m_rf.score(x_train, y_train))
    score_test.append(m_rf.score(x_test, y_test))
    
plt.plot(np.arange(1,101), score_train, label='train_score')
plt.plot(np.arange(1,101), score_test, label='test_score', color='red')
plt.legend()

# 3-2) 적절한 split 수 선택 (min_samples_split) : 11
score_train=[]; score_test=[]
for i in np.arange(2,21):
    m_rf=rf(min_samples_split=i)
    m_rf.fit(x_train, y_train)
    score_train.append(m_rf.score(x_train, y_train))
    score_test.append(m_rf.score(x_test, y_test))
    
plt.plot(np.arange(2,21), score_train, label='train_score')
plt.plot(np.arange(2,21), score_test, label='test_score', color='red')
plt.legend()

# 3-3) 적절한 features 수 선택 (max_feature)!!다시보기!!
score_train=[]; score_test=[]
for i in np.arange(1, df_cancer['data'].shape[1] + 1):
    m_rf=rf(max_features=i)
    m_rf.fit(x_train, y_train)
    score_train.append(m_rf.score(x_train, y_train))
    score_test.append(m_rf.score(x_test, y_test))


plt.plot(np.arange(1,31), score_train, label='train_score')
plt.plot(np.arange(1,31), score_test, label='test_score', color='red')
plt.legend()

#4. 최종 모델 고정
m_rf = rf(max_features=11 , min_samples_split=11, n_estimators=10)
m_rf.fit(x_train, y_train)
m_rf.score(x_test, y_test)

#5. 특성 중요도 시각화 !! 다시보기 !!

def plot_feature_importances(model, data):
    n_features = data.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    ply.yticks(np.arange(n_features), data.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)
    
plot_feature_importances(rf,df_cancer)

# =============================================================================
# Gradiant Boosting Tree
# =============================================================================
 - 여러개의 결정 트리를 묶어 강력한 모델을 만드는 또 다른 앙상블 방법
 - 이전 트리를 학습하는 형태
 - learning_rate로 정해진 학습률에 따라 오분류 데이터 포인트에 더 높은 가중치 부여,
   => 다음 생성되는 트리는 오분류 데이터의 올바른 분류에 초점을 두는 형식
 - 복잡도가 낮은 초기 트리 모델로부터 점차 복잡해지는 형태를 갖춤
 - 랜덤포레스트보다 저 적은 수의 트리로 높은 예측력 기대 가능
 - 각 트리는 서로 독립적일 수 없으므로 n_jobs 같은 parallel 옵션에 대한 기대가 줄어든다.
 


# =============================================================================
# # 참고 : extreme Gradiant Boosting Tree(XGB) 설치 및 로딩 방법
# =============================================================================
# GB : 앞 트리의 학습 결과를 활용해 재보정 후 새로운 트리 생성 (learning mate - 새로운 매개변수)
# - XGB : 빠른 속도록 트리구조 구성
# - pip 파이썬 설치 명령어 (파이썬 홈페이지에서 설치)
# - conda 아나콘다 설치 명령어 (아나콘다에서 설치)

anconda search -t conda xgboost  # 채널 찾는 명령어

conda install py-xgboot  # 아나콘다 채널로 설치 * 파이썬에 등록되지 않은 경우 사용
pip install xgboost      # os 명령어

# 설치 및 로딩
pip install xgboost      # 파이썬으로 설치 - 사용
from xgboost.sklearn import XGBClassifier as xgb
from xgboost.sklearn import XGBRegressor as xgb_r
# =============================================================================










