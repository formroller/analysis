# =============================================================================
# # 교호작용 기법(interaction)
# =============================================================================
 - 변수간의 결합*
 - 설명변수끼리의 상호곱 변수 생성, 그 중 의미있는  변수를 추출하기 위함
 - 적은 설명변수로 많은 설명변수로의 확장 가능 (설명변수 추가 없이)
 - 초기 변수 연구 단계에서 고려되는 기법
 => 수업 목표 : "의미있는 교호작용(변수간 결합)을 찾는 것이 목적"
 
아파트 가격 <= 지하주차장 면적 * 평균강수량
아파트 가격 <= 지하주차장 면적 + ... +

run profile1
from sklearn.preprocessing import PolynomialFeatures

# interaction - iris data
# 1.데이터 로딩
df_iris = iris()

# 2.데이터 분리
train_x,test_x,train_y,test_y = train_test_split(df_iris['data'],
                                                 df_iris['target'],
                                                 random_state=0)

# 3. interaction 모델 생성
m_poly = PolynomialFeatures(degree=2)  # degree=2, 2차원

# 4. 모델에 데이터 학습 -> 설명변수 변환기준 찾기
 #fitting : (확장된 모델알려준다)변수명 추출
m_poly.fit(train_x)

# 5. 실 데이터 변환 (교호작용 기준에 맞게 설명변수 변환)
# transform : 확장된 형식에 맞춰 리폼(실데이터에 대한 변환)
train_x_poly = m_poly.transform(train_x)

# 6. 변환된 변수 형태 확인
m_poly.get_feature_names()                           # 실제 변수명X
m_poly.get_feature_names(설명변수)                   # 설명변수명 출력  
m_poly.get_feature_names(df_iris['feature_names'])  # 실제 변수명으로 추출

['sepal length', 'sepal width', 'peral length', 'petal width']

            fitting
y  x1  x2   =>   y x1  x2  x1^2  x2^2  x1x2
    1   2           1   2    1     4     2 
    2   3           2   3    4     9     6 
   
transform : 확장된 형식에 맟줘 리폼(실데이터에 대한 변환)
fitting : (확장된 모델알려준다)변수명 추출


# 7. 확장된 데이터셋으로 knn모델 적용
m_knn=knn(n_neighbors=3)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)  # 4개 설명 변수 예측값 - 97.37%

m_knn2=knn(n_neighbors=3)
m_knn2.fit(train_x_poly, train_y)
test_x_poly = m_poly.transform(test_x)
m_knn2.score(test_x_poly, test_y)  # 확장된 설명변수 예측값 - 97.37%

# 8. 확장된 설명변수 중 의미있는 교호작용 추출

m_rf = rf()
m_rf.fit(train_x_poly, train_y)

df_iris_poly_col = m_poly.get_feature_names(df_iris['feature_names'])
s1 = Series(m_rf.feature_importances_, index = df_iris_poly_col)
s1.sort_values(ascending=False)

# =============================================================================
# #[ 연습 문제 : interaction - cancer data]
# =============================================================================



#------------------
#1) 데이터 로딩
df_cancer = cancer()
#2) 데이터 분리
train_x,test_x,train_y,test_y = train_test_split(df_cancer['data'],
                                                 df_cancer['target'],
                                                 random_state=0)
#3) interaction 모델 생성
m_poly = PolynomialFeatures(degree=2)
#4) 모델에 데이터 학습
m_poly.fit(train_x) # 어떤 형태의 설명변수가 들어가도 상관 없다
#5) 실 데이터 변환(2차항 추가 / 변수 형태 변경)
train_x_poly = m_poly.transform(train_x)
test_x_poly = m_poly.transform(test_x)
#6) 변환된 변수 형태 확인
m_poly.get_feature_names(df_cancer['feature_names'])
#7) 확장된 데이터셋으로 knn모델 적용 -> scale에 민감 / 고차원 데이터셋(설명변수 많음)에 불리
# RF로 다시 해보기**
#7-1) 원본 데이터 셋 적용
m_knn = knn(n_neighbors=3)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)   # 기존 변수 예측값 = 92.3%
#7-2) 확장된 데이터 셋 적용
m_knn2 = knn(n_neighbors=3)
m_knn2.fit(train_x_poly, train_y)
m_knn2.score(test_x_poly, test_y)  # 확장된 설명 변수 예측값  = 93.7% (1.4% 증가)
#8) 확장된 설명변수 중 의미있는 교호작용 추출(특성 중요도 파악)
m_rf = rf()
m_rf.fit(train_x_poly, train_y)

df_cancer_poly_col = m_poly.get_feature_names(df_cancer['feature_names'])
s1 = Series(m_rf.feature_importances_, index = df_cancer_poly_col)
v1_importance = s1.sort_values(ascending=False)[:30]

#9) 시각화
plt.barh(range(30), v_importance, align = 'center')
plt.yticks(np.arange(30), v_importance.index)

랜덤포레스트(max_features) - 후보군의 갯수 낮아도 문제 발생할 수 있다. 
=> 해당 데이터에서 max_features 튜닝시 높은 예측력 갖을 수 있다.


# =============================================================================
# 변수 스케일링
# =============================================================================
 - 주로 설명변수끼리의 범위를 비슷하게 조절하기 위해 사용
 - 거리기반 모델, NN모델 등에서 모델 학습 전 수행되어야 한다.
 - 설명변수의 정확한 변수 중요도를 측정하기 위해 필요.
 - interaction 고려시 차원이 더 높은 변수가 설명력 높게 나오는 현상을 보완

 - 거리와 관계된 모든 모델은 scale 필요 (0~1 or -1~1)
 - Standard / MinMax scaler
 의미있는 교호작용 set 찾기위해 
 - 표준화
 => 표준화 없이 interaction시 제곱 할수록 해당 항이 유의하다고 나오는 문제 발생. 
 
 
# 1) MinMax scaler
 - 각 설명변수마다 최솟값에 0, 최댓값에 1을 부여하도록 재조절하는 방식
 * 모든 스케일 값은 0 또는 1 사이에 배치
 * 최댓값에 1 부여 -> 1값 기준으로 다른 값 나누기
 * 절대 음의 값이 나올 수 없다.
 fit - min / max 찾음
from sklearn.preprocessing import MinMaxScaler
 
# 2) Standard scaler
 - 각 설명변수마다 표준화 시키는 방식
 - 표준화 : (x - xbar) / s 
 * xbar : 표본평균 
 * s : 표본표준편차
 fit - xbar / s 찾음
from sklearn.preprocessing import StandaradScaler 
 
 

# minmax scaling - iris data
# 1. 데이터 로딩 및 분리
df_iris = iris()
train_x, test_x, train_y, test_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state=0) 

# 2. scaling model 생성
m_mms = MinMaxScaler()
m_mms.fit(train_x)                          # 각 설명변수의 최대/최소 확인만 *check
train_x_scaled = m_mms.transform(train_x)   # 최대/최소에 맞게 데이터 변환   *transform
test_x_scaled = m_mms.transform(test_x) #test set 이나 train 기준으로 해석(1)

# 변환된 각 설명변수의 최대 / 최소 확인
train_x_scaled.min(axis=0) 
train_x_scaled.max(axis=0) 
 => 0 / 1 사이
 
test_x_scaled.min(axis=0) 
test_x_scaled.max(axis=0) 
 => 0/1 기준으로 출력되지 않음.
 => fit(train_x)  > fit(test_x)로 적용해야 test 0과 1로 출력된다
 
# 참고 : train, test data set마다 서로 다른 기준으로 scaling 하는 경우
m_mms2 = MinMaxScaler()
m_mms2.fit(test_x)                          # 각 설명변수의 최대/최소 확인만 *check
test_x_scaled2 = m_mms.transform(test_x)    #test set 이며 test기준으로 해석(2)


test_x_scaled2.min(axis=0) 
test_x_scaled2.max(axis=0) 
 
test_x_scaled = m_mms.transform(test_x) #test set 이나 train 기준으로 해석(1)
test_x_scaled2 = m_mms.transform(test_x)    #test set 이며 test기준으로 해석(2)
#=> scale은 (1)로 수행 - 기준이 달라지면 안되기 때문!**
# 수치만 작을 뿐 포인트간의 거리 유지해야한다(원본의 분포를 유지해야 좋은 scaling)**
# 왜곡이 없어야 한다.


#[ 참고 : scale된 각 데이터의 분포 확인 ] 
 # 올바른 scaling 작업이라면 원본의 분포와 일치할 것!

fig, axes = plt.subplots(1,3)

.scatter : x, y 2차원 형식 산점도 
#1) 원본데이터의 산점도
#- 옳바른 스케일링 (서로 같은 기준)
axes[0].scatter(train_x[:,0],train_x[:,1],
                c=mglearn.cm2(0), label='train', s=60)  # train 산점도 [cm2(색상) - 팔렛트  | s= 포인트 크기]

axes[0].scatter(test_x[:,0],test_x[:,1],
                c=mglearn.cm2(1), label='test', s=60)   # test 산점도  

axes[0].legend()
axes[0].set_title('raw data')


#2) 올바른 스케일링 데이터의 산점도(train, test 서로 같은 기준으로 조절)
axes[1].scatter(train_x_scaled[:,0],train_x_scaled[:,1],
                c = mglearn.cm2(0), label='train', s=60) 

axes[1].scatter(test_x_scaled[:,0], test_x_scaled[:,1],
                c = mglearn.cm2(1), label='train', s=60) 
 
axes[1].legend()
axes[1].set_title('correct scale')

=> 분포는 완벽하게 동일

#3) 잘못된 스케일링 데이터의 산점도 (train, test가 서로 다른 기준으로 조절)
axes[2].scatter(train_x_scaled[:,0],train_x_scaled[:,1],
                c = mglearn.cm2(0), label='train', s=60) 

axes[2].scatter(test_x_scaled2[:,0], test_x_scaled2[:,1],
                c = mglearn.cm2(1), label='train', s=60) 
 
axes[2].legend()
axes[2].set_title('incorrect scale')

# 각각 서로다른 기준으로 스케일링시 데이터 왜곡될 수 있다.

# 스케일링 조절된 형태의 의미있는 interaction 추출 및 모델 적용

# 스케일링 전)
df_cancer=cancer()

train_x,test_x,train_y,test_y = train_test_split(df_cancer.data,
                                                 df_cancer.target,
                                                 random_state=0) 
m_poly = PolynomialFeatures(degree=2)
m_poly.fit(train_x)

train_x_poly = m_poly.transform(train_x)
cancer_poly_col = m_poly.get_feature_names(df_cancer.feature_names)

m_rf = rf()
m_rf.fit(train_x_poly, train_y)
m_rf.score(test_x_poly,test_y)    # 90.6%

# 스케일링 후) * 스케일링 후 interaction 수행
df_cancer=cancer()

train_x,test_x,train_y,test_y = train_test_split(df_cancer.data,
                                                 df_cancer.target,
                                                 random_state=0) 
#-- 스케일링
m_ms = MinMaxScaler()
m_ms.fit(train_x)
train_x_sc = m_ms.transform(train_x)
test_x_sc = m_ms.transform(test_x)

train_x_sc.min(axis=0)
train_x_sc.max(axis=0)

test_x_sc.min(axis=0)
test_x_sc.max(axis=0)

# 
m_poly = PolynomialFeatures(degree=2)
m_poly.fit(train_x_sc)
train_x_poly = m_poly.transform(train_x_sc)
test_x_poly = m_poly.transform(test_x_sc)

cancer_poly_col = m_poly.get_feature_names(df_cancer.feature_names)


# feature importance 확인
m_rf = rf()
m_rf.fit(train_x_poly,train_y)
m_rf.score(test_x_poly, test_y)  # 91.1% 

s1 = Series(m_rf.feature_importances_, index = cancer_poly_col)
v_importance = s1.sort_values(ascending = False)[:30]

plt.barh(range(30), v_importance, align='center')
plt.yticks(np.arange(30), v_importance.index)

# [연습 : 독버섯 데이터 셋의 분류분석 ]
• https://archive.ics.uci.edu/ml/datasets/Mushroom 에서 데이터 설명 확인 가능 
• 첫 번째 열이 독성의 유무로 독성이면 p, 식용이면 e로 표현 
• 두 번째 열은 버섯의 머리 모양     (벨형태 : b, 원뿔 : c, 볼록한 형태 : x, 평평한 형태 : f, 혹 형태 : k,오목한 형태 : s) 
• 네 번째 열은 버섯의 머리 색      (갈색 : n, 황갈색 : b, 연한 갈색 c .....) 

mr = pd.read_csv('mushroom.csv')

# SVM : Support Vector Machine(커널 서포트 벡터 머신)
 - 분류분석 모델
 - 다차원 데이터셋에 주로 적용
 - 다차원 데이터셋의 분류 기준을 초평면으로 만들어 분류하는 과정
 - 초평면을 만드는 과정이 매우 복잡, 해석 불가(black box 모델)
 - c, gamma의 매개변수 조합이 매우 중요
 - 학습전 scaling 조절 필요
 - 이상치에 민감
 - 오분류 데이터에 가중치를 수정해 
   선형-> 비선형/ 저차원 -> 다차원 판 형태로 분류기준 강화시킴
