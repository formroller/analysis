# 데이터 분석
데이터를 통해 숨겨진 의미가 있는 정보를 도출하는 과정 => 미래 예측

ex)
 - 주가 / 날씨 / 수요 예측
 - 암 종양의 양성 / 악성 예측
 - 고객의 이탈 예측
 - 고객의 장바구니 예측 및 추천
 
# [데이터 분석의 종류]
1. 지도학습
 - Y값 존재
 - 관심있는 Y값을 X로 설명/예측 하는 작업
 1) 회귀 : Y - 연속형
 2) 분류 : Y - 범주형

2. 비지도 학습 
 - Y값 없음
 - 데이터로부터 유사성에 근거한 군집을 분류
 - 비슷한 특징을 갖는 데이터의 분류
 
# [데이터 분석 과정]
 1. 목적 (따릉이 고장여부)
 2. 데이터 수집(자전거 상세데이터, 날씨 데이터, ...)
 3. 모델 선택
 4. 변수 연구(Y~X) : 변수의 선택 (feature selection)
                     변수의 가공
                     변수 결합
 5. 예측 수행(학습)
 6. 모델 평가
 7. 튜닝
 8. 결과 해석 및 비즈니스 모델 적용
 
# 데이터 분석에 필요한 라이브러리(모듈)
 - scikit-learn  : 파이썬 분석 모듈(알고리즘, 데이터 제공)
 - numpy, pandas : 정형, 비정형 데이터 담기 위한 모듈
 - scipy         : 복잡한 수학 연산
 - matplotlib    : 그래프 출력
 - mglearn       : 복잡한 시각화 함수 제공(팔레트 등) ,
                   외부 모듈로 아나콘다 미포함 => 설치 필요

import mglearn

# sklearn에서의 데이터 셋 형태
from sklearn.datasets import load_iris
df_iris = load_iris()  # 딕셔너리 형태
type(df_iris)          # sklearn.untils.Bunch

df_iris.keys()           # ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
df_iris['data']          # 설명변수(독립변수)의 집합
df_iris['target']        # 종속변수 데이터, 숫자형 범주 데이터로 표현
df_iris['target_names']  # 종속변수 실제 이름
df_iris['feature_names'] # 설명변수의 각 이름

# =============================================================================
# #knn
# =============================================================================
 - 지도학습 > 분류분석 > 거리기반 모델
 - 예측하고자 하는 데이터와 기존 데이터 포인트(각 관측치)들위 거리가 가까운
   k개의 이웃이 갖는 정답(Y)의 평균 및 다수결로 Y를 예측하는 형태
 - 이상치(outlier)에 매우 민감 => 제거 혹은 수정 필요
 - 설명변수의 scale에 매우 민감 => scale 조정 필요 (표준화)
 - 모델 학습시 선택된 설명변수의 조합에 따라 매우 다른 예측력을 보임
 (모델 자체 feature selection 기능 없음)

d=sqrt((x11 - x12)^2 + (x21 - x22)^2 - (x31 - x32)^2 + (x41 - x42)^2)

# [분류분석 절차(knn)]

#1. 데이터 분류(train / validation/ test set)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,  # 설명변수 데이터 셋
                                                    y,  # 종속변수 데이터 셋
                                                    train_size=0.75, # train set 비중(기본값-75%)
                                                    random_state=0)  # seed 값 고정

x_train, x_test, y_train, y_test = train_test_split(df_iris['data'],
                                                    df_iris['target'],
                                                    random_state=99)
x_train.shape  # (112, 4) => trunc(150*75)의 row 추출
x_test.shape   # (39, 4)

#2. 데이터 학습
from sklearn.neighbors import KNeighborsClassifier as knn

m_knn = knn(n_neighbors=2)
m_knn.fit(x_train, y_train)

#3. 모델 평가
m_knn.score(x_test, y_test)  # 92.1

#[참고 : score 메서드 사용하지 않고 평가 점수 얻는 방법 (R과 유사)]
m_knn.predict(x_test)  # test set에 대한 예측 결과
y_test                 # test set에 대한 실제값 
sum(y_test == m_knn.predict(x_test)) / y_test.shape[0] * 100

#4. 매개변수 튜닝
#4-1) 반복문을 통한 k값의 변화에 따른 train, test set의 계산 score
score_train=[];score_test=[];

for i in np.arange(1,11):
    m_knn = knn(n_neighbors=i)
    m_knn.fit(x_train, y_train)
    score_train.append(m_knn.score(x_train,y_train))
    score_test.append(m_knn.score(x_test, y_test))
    
#4-2) 시각화
import marplotlib.pyplot as plt
plt.plot(np.arange(1,11), score_train, label='train_score')
plt.plot(np.arange(1,11), score_test, label='test_score', color='red')

plt.legend()

#5. 예측
new_data = np.array([5.5, 3.0, 2.5, 0.9])
new_data2 = np.array([[5.5, 3.0, 2.5, 0.9]])

m_knn.predict(new_data)   # error, input 데이터 2차원 형태 아님
m_knn.predict(new_data2)  # 정상 수행

df_iris['target_names'][m_knn.predict(new_data2)][[0]]  # array(['setosa'])
df_iris['target_names'][m_knn.predict(new_data2)][0]    # 'setosa'

# knn - 회귀
 - knn은 기본이 분류 모델이나 회귀 모델도 제공
 - 전통 회귀와는 다르게 인과관계 분석보단 예측에 초점을 맞춘 알고리즘
 - 전통 회귀의 여러 통계적 가설(오차의 정규성, 등분산성, ...)을 따르지 않아도 무방
 
 [참고 - R^2]
 - knn 회귀 모델 평가는 R^2로 제공
 - R^2는 총 분산을 회귀식으로 얼마나 설명했는가를 의미
 
 (y - ybar) = (y - yhat) + (yhat - ybar)
 
sum((y - ybar)^2) = sum((y - yaht)^2) + sum((yhat - ybar)^2)
SST(총 편차의 합) =    SSE(오차제곱합) + SSR(회귀제곱합)
      총분산      =          MSE      +    MSR
      
R^2 = SSR / SST = 1 - SSE / SST


# =============================================================================
# #[ 교호작용(상호작용) - interaction ]
# =============================================================================
아래와 같은 설명 변수를 갖는 경우 각 설명변수끼리의 상호작용 고려
x1 x2 x3

1) 2차항 고려 : x1^2, x2^2, x3^2, x1x2, x1x3, x2x3
2) 3차항 고려 : x1^2, x2^2, x3^2, x1x2, x1x3, x2x3,
                x1^3, x2^3, x3^3, x1^2x2, ...., x2x3^2, x1x2x3
