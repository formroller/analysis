# SVM : Support Vector Machine(커널 서포트 벡터 머신)
 - 분류분석 모델
 - 다차원 데이터셋에 주로 적용
 - 다차원 데이터셋의 분류 기준을 초평면으로 만들어 분류하는 과정
 - 초평면을 만드는 과정이 매우 복잡, 해석 불가(black box 모델)
 - c, gamma의 매개변수 조합이 매우 중요
 - 학습전 scaling 조절 필요**
 - 이상치에 민감**
 - 오분류 데이터에 가중치를 수정해 
   선형-> 비선형/ 저차원 -> 다차원 판 형태로 분류기준 강화시킴
 - np.hstack = concat
 - [:,1:] * 차원 축소 방지
 - 고차원 데이터셋에 적합
 
 [ 매개변수 ]
 - c : 비선형을 강화하는 매개변수  (분류선)
 - gamma : 차원의 확장(분류판). 
         => 감마의 변화에 따라 시각화 할 경우 차원이 확장된다.
 - 매개변수 간 연관성 있음

     
    
#교제 예제 P.91
# 임의 데이터로부터 이차항을 추가, 3차원 측면에서의 데이터 분리 초평면 유도 및 시각화
run profile1
from sklearn.svm import SVM

# SVM - cancer data
#1. 데이터 로딩 및 분리
df_cancer = cancer()
train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

#2. 모델 생성 및 학습
m_svm = SVC()
m_svm.fit(train_x, train_y)

#3. 모델 평가
m_svm.score(test_x, test_y)  # 62.94% -> 예측력 낮다. >> scaling 해야한다.

#스케일링)
df_cancer = cancer()
train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)
m_ms = MinMaxScaler()
m_ms.fit(train_x)
train_x_sc = m_ms.transform(train_x)
test_x_sc = m_ms.transform(test_x)

m_svm = SVC()
m_svm.fit(train_x_sc, train_y)
m_svm.score(test_x_sc, test_y)   # 95.1%

#4. 매개변수 튜닝
v_c = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]
v_gamma = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

col_c=[];col_gamma=[];col_tr_score=[];col_te_score=[]
for i in v_c :                              #* best score (최적의 조합 찾아야한다)
    for j in v_gamma : 
        m_svm = SVC(C = i, gamma=j)
        m_svm.fit(train_x_sc, train_y)
        col_c.append(i)
        col_gamma.append(j)
        col_tr_score.append(m_svm.score(train_x_sc, train_y))
        col_te_score.append(m_svm.score(test_x_sc, test_y))

df_svm_score = DataFrame({'C':col_c,
                          'gamma':col_gamma,
                          'tr_score':col_tr_score,
                          'te_score':col_te_score})         

# best score 및  매개변수 조합 확인 (정렬로 부터)
df_score = df_svm_score.sort_values(by=['tr_score','te_score'], ascending=False)  # tr기준 100/96.5
df_score1 = df_svm_score.sort_values(by=['te_score','tr_score'], ascending=False) # te 기준 -> 98.6/98 (우수)


=> gamma 점수 올린다면 모델이 복잡해진다(overfit현상 발생) -> 새로운 데이터에 대한 예측력 저하
=> 정렬시키며 reindex현상 발생

# best score 및 매개변수 조합 시각화
df_score.iloc[:,2:4].plot()  # 인덱스 순서 꼬임현상

df_score.index = np.arange(df_score.shape[0])  # index 조정
df_score.iloc[:,2:4].plot()

# x축에 전달한 매개변수 조합 결합
# c, gamma 결합해 표시  
ex) c:100.0   gamma:1.000
      100.0         1.000 
      
f1 = lambda x, y : 'C: ' + str(x) + ' gamma: ' + str(y)

df_score.C.map(f1,df_score.gamma)         # map 메서드는 추가 인자 전달 불가
list(map(f1, df_score.C, df_score.gamma)) # map 함수 적용 가능(df에서 여러개 인자 전달)

f2 = lambda x : 'C: ' + str(x[0]) + ' gamma: ' + str(x[1])  # apply는 하나의 벡터(리스트)묶어 전달
v_text = df_score.apply(f2,axis=1)

plt.xticks(df_score.index,   # 첫 인자 : 분할 가능한 숫자벡터 들어가야한다.
           v_text,           # 눈금별 이름
           rotation = 90,
           fontsize = 5)           

# [참고]
# plt.xticks = axis (x축 눈금 정의 함수)
# axis(1, at=1:46, label=c('a','b',...))  in R


# test score에 대한 히트맵 출력 (7*7 배열 생성)  !! 다시보기!!
df_svm_score  # c고정, gamma 늘어나는 형식

import mglearn
mglearn.tools.heatmap(values,         #  농도를 표현할 2차원 숫자배열
                      xlabel,         # x축 이름  
                      ylabel,         # y축 이름
                      xticklabels,    # x축 눈금
                      yticklabels,    # y축 눈금
                      cmap=None)      


a1 = np.array(df_svm_score.te_score).reshape(7,7)
mglearn.tools.heatmap(a1, 'gamma', 'C', v_gamma, v_c, cmap='viridls')  # gridsearch 기법 - c/gamma 동시 해석

# =============================================================================
#  PCA : 주성분 분석 (비지도 학습)
# =============================================================================
 - 비지도학습 : train 시킬 데이터에 Y없음.
 - 목적 : 변수의 선형 결합, 그로인한 변수 축소 => 모델의 복잡도 제어
 - 회귀에서의 다중공선성의 해결 방안이기도 함
 - 다차원(=고차원) 데이터의 차원 축소로 인한 과대적합 해소
 - 기존 변수의 결합으로 의미있는 새로운 변수 유도(분산으로)**
 
 => PCA가 최종 모델로 Y를 유도할 수 없다 (중간 단계 변수 축소 결합 목적)**
 => 분산 크다 -> 중요도 높다(높은 가중치 부여) => 이미지 예측률 높아진다.
 => 첫번째 유도된 인공변수(c1)가 전체 분산을 유도.
 => 유도된 변수간(c1, c2) 독립적
 => 정보를 유지하며 차원을 축소한다. ex) c1 = x1 +x2 +x3
  => 회귀분석시 설명변수 중복된 경우 계수 추정 불가할 수 있다(다중공선성)
  => 따라서 PCA 기법 사용해 다중공선성 방지할 수 있다.(유도된 변수간 독립적이기 때문)
 
 PCA + SVM
 PCA + KNN (이미지 분석)
 
 # PCA + knn - iris data
 => 중간모델 + (최종모델)
 - 스케일링 후 PCA에 적용해야 동일한 기준(분산)하에 비교할 수 있다.

from sklearn.decomposition import PCA

# 1. 데이터 로딩
df_iris = iris()
iris_tr_x, iris_te_x, iris_tr_y, iris_te_y = train_test_split(df_iris.data,
                                                              df_iris.target,
                                                              random_state=0)


# 2. scaling
m_sc = StandardScaler()
m_sc.fit(iris_tr_x)
train_x_sc = m_sc.transform(iris_tr_x)
test_x_sc= m_sc.transform(iris_te_x)

# 3. PCA로 2개 인공변수 유도
# c1 = ax1 + bx2 + cx3 + dx4 ()
# c2 = ax1 + bx2 + cx3 + dx4 ()

# 3-1) 인공변수 유도
m_pca = PCA(n_components=2)
m_pca.fit(train_x_sc)  # data학습시켜 적절한 계수 추정하는 것이 목적

m_pca.components_  # 계수 출력
array([[ 0.53429768, -0.2166997 ,  0.58431023,  0.57109438],  # c1 
       [ 0.33124095,  0.94253335,  0.00413573,  0.04351097]]) # c2

# 3-2) 인공변수 변환
train_x_sc_pca = m_pca.transform(train_x_sc) # c1, 두개의 변수로 변환
test_x_sc_pca = m_pca.transform(test_x_sc)   # c2, 

# 3-3) 유도된 인공변수 분포 시각화
mglearn.discrete_scatter(train_x_sc_pca[:,0],
                         train_x_sc_pca[:,1],
                         iris_tr_y)

# 4. knn 모델 적용 (final model)
m_knn = knn(n_neighbors= 3)
m_knn.fit(train_x_sc_pca, iris_tr_y)
m_knn.score(test_x_sc_pca, iris_te_y)  # 89.47% ()
# knn만 적용시 보다 낮은 결과 도출
# -> 모델 단순화해 예측력 낮을 수 있다.

# =============================================================================
# PCA + SVM - cancer data
# =============================================================================
1. svm
2. pca + svm

#1. SVM
cancer = cancer()
cancer_tr_x, cancer_te_x, cancer_tr_y, cancer_te_y  = train_test_split(cancer.data,
                                                                       cancer.target,
                                                                       random_state=0)


m_ms = MinMaxScaler()
m_ms.fit(cancer_tr_x)

cancer_tr_x_sc = m_ms.transform(cancer_tr_x)
cancer_te_x_sc = m_ms.transform(cancer_te_x)

cancer_tr_y.shape

m_svm = SVC()
m_svm.fit(cancer_tr_x, cancer_tr_y)
m_svm.score(cancer_te_x_sc, cancer_te_y)  # 62.93

# -- 매개변수 튜닝
v_c = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]
v_gamma = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

col_c=[]; col_gamma=[]; col_tr_score=[]; col_te_score=[]
for i in v_c:
    for j in v_gamma:
        m_svm = SVC(C=i, gamma=j)
        m_svm.fit(cancer_tr_x_sc, cancer_tr_y)
        col_c.append(i)
        col_gamma.append(j)
        col_tr_score.append(m_svm.score(cancer_tr_x_sc, cancer_tr_y))
        col_te_score.append(m_svm.score(cancer_te_x_sc, cancer_te_y))
        
df_svm_score = DataFrame({'C':col_c,
                          'gamma':col_gamma,
                          'train_score':col_tr_score,
                          'test_score':col_te_score})

df_score = df_svm_score.sort_values(by = ['test_score','train_score'], ascending = False)
df_score.index = np.arange(df_score.shape[0])    

# 2. PCA + SVM
m_pca = PCA(n_components = 2)
m_pca.fit(cancer_tr_x_sc)

m_pca.components_

train_x_sc_pca = m_pca.transform(cancer_tr_x_sc)
test_x_sc_pca = m_pca.transform(cancer_te_x_sc)

mglearn.discrete_scatter(train_x_sc_pca[:0],
                         train_x_sc_pca[:1],
                         cancer_tr_y)

m_svm = SVC()
m_svm.fit(train_x_sc_pca, cancer_tr_y)
m_svm.score(test_x_sc_pca, cancer_te_y)  # 92.3%

# =============================================================================
# PCA 시각화(grid search) - cancer data
# =============================================================================

#1. 데이터 로딩 및 분리
df_cancer = cancer()
train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

#2. 모델 생성 및 학습
m_svm = SVC()
m_svm.fit(train_x, train_y)

#3. 모델 평가
m_svm.score(test_x, test_y)  # 62.94% -> 예측력 낮다. >> scaling 해야한다.

#스케일링)
df_cancer = cancer()
train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)
m_ms = MinMaxScaler()
m_ms.fit(train_x)
train_x_sc = m_ms.transform(train_x)
test_x_sc = m_ms.transform(test_x)

m_svm = SVC()
m_svm.fit(train_x_sc, train_y)
m_svm.score(test_x_sc, test_y)   # 95.1%

#4. 매개변수 튜닝
v_c = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]
v_gamma = [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]

col_c=[];col_gamma=[];col_tr_score=[];col_te_score=[]
for i in v_c :                              #* best score (최적의 조합 찾아야한다)
    for j in v_gamma : 
        m_svm = SVC(C = i, gamma=j)
        m_svm.fit(train_x_sc, train_y)
        col_c.append(i)
        col_gamma.append(j)
        col_tr_score.append(m_svm.score(train_x_sc, train_y))
        col_te_score.append(m_svm.score(test_x_sc, test_y))

df_svm_score = DataFrame({'C':col_c,
                          'gamma':col_gamma,
                          'tr_score':col_tr_score,
                          'te_score':col_te_score})         

import mglearn
a1 = np.array(df_svm_score.te_score).reshape(7,7)
mglearn.tools.heatmap(a1, 'gamma', 'C', v_gamma, v_c, cmap='viridls')  # gridsearch 기법 - c/gamma 동시 해석
# =============================================================================
