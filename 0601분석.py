run profile1

# 원 핫 인코딩
- 문자변수를 0과 1의 값을 갖는 이진 데이터로 변경
- 모델에 따라 문자값의 학습을 금지하는 경우 주로 사용
- 딥러닝 모델에서의 Y값은 원 핫 인코딩이 필수

# 예제) 아래 데이터프레임에서 col1의 컬럼 값을 이진 데이터로 표현
df1 = DataFrame({'col1':['M','M','F'],
                 'col2':[90,89,93]})

pd.get_dummies(df1)
pd.get_dummies(df1, drop_first=True)
pd.get_dummies(df1, columns=['col1','col2'])

# feature selection
- 모델 학습 전 모델에 학습시킬 변수를 선택하는 과정
- 모델 자체가 변수 선택 알고리즘을 포함하는 경우도 존재
- 회귀, 거리기반 모델들은 모델에 학습시킬 변수의 조합에 따라
  매우 예측력이 달라지는 모델이므로 변수 선택의 중요성이 큼
- 트리기반, 신경망 모델들은 모델 자체에서 변수 선택이 어느정도 이뤄지므로
  다른 모델에 비해 사전 변수 선택의 중요도가 낮음
  
1) 모델에 의한 변수 선택(모델 기반 변수 선택법)
 - 트리기반, 회귀기반 모델에서의 변수 중요도를 참고
 - 트리기반에서는 변수 중요도(feature importance)를, 
   회귀에서는 각 변수의 계수를 참고하여 사용
 - 모델에 학습된 변수의 상관 관계도 함께 고려  
 
# iris data feature selection - 모델 기반 변수 선택법
from sklearn.feature_selection import SelectFromModel
 
# 1. 데이터 로딩 
df_iris = iris()

# 2. 불필요한 변수 추가   
vseed = np.random.RandomState(0) 
vcol  = vseed.normal(size=(len(df_iris.data), 10)) # noise 변수 10개 추가

# 3. 기존 데이터와 결합
np.hstack([df_iris.data, vcol]).shape  # 150 rows, 14 columns 확인
df_iris_new = np.hstack([df_iris.data, vcol])

# 4. 랜덤포레스트에 의한 모델 기반 변수 선택 
m_rf = rf()
m_fs2 = SelectFromModel(m_rf, threshold='median')
m_fs2.fit(df_iris_new, df_iris.target)
m_fs2.get_support()

# 5. 각 변수 중요도 출력
m_fs2.estimator_.feature_importances_

# [ 연습 문제 ]
# cancer data에서 상위 50%의 설명변수 선택, 선택 전/후 모델을 SVM으로 비교

2) 일변량 통계 기법
 - 각 변수 하나하나를 독립적으로 각 변수 중요도를 계산
 - 학습 시킬 모델이 필요 없어 연산 속도가 매우 빠름
 - 각 변수 마다의 종속변수와의 상관 관계 파악

# iris data feature selection - univariate statistic(일변량)
# 1. 데이터 로딩
df_iris = iris()

# 2. 불필요한 변수 추가   
vseed = np.random.RandomState(0) 
vcol  = vseed.normal(size=(len(df_iris.data), 10)) # noise 변수 10개 추가

# 3. 기존 데이터와 결합
np.hstack([df_iris.data, vcol]).shape  # 150 rows, 14 columns 확인
df_iris_new = np.hstack([df_iris.data, vcol])

# 4. 변수 선택
from sklearn.feature_selection import SelectPercentile
m_fs1 = SelectPercentile(percentile=30)  # 모델 생성
m_fs1.fit(df_iris_new, df_iris.target)   # 각 설명변수와 종속변수의 상관관계 파악
m_fs1.get_support()                      # 선택 변수 확인

m_fs1.transform(df_iris_new)             # 선택된 변수 집합으로 변환

# 5. 선택된 변수 시각화
plt.matshow(m_fs1.get_support().reshape(1,-1), cmap='gray_r')

# [ 연습 문제 ]
# cancer data에서 상위 50%의 설명변수 선택, 선택 전/후 모델을 SVM으로 비교
# 1. data loading
df_cancer = cancer()

# 2. 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(df_cancer.data,
                                                    df_cancer.target,
                                                    random_state=0)

# 3. 변수 선택
m_fs1 = SelectPercentile(percentile=50) 
m_fs1.fit(train_x, train_y)
m_fs1.get_support()         

# 4. 선택된 변수 확인
df_cancer.feature_names[m_fs1.get_support()]

# 5. 모델 학습
# 5-1) 변수 선택 전
m_svm1 = SVC()
m_svm1.fit(train_x, train_y)
m_svm1.score(test_x, test_y)   # 62.9

# 5-2) 변수 선택 후
train_x_selected = m_fs1.transform(train_x)
test_x_selected  = m_fs1.transform(test_x)

m_svm2 = SVC()
m_svm2.fit(train_x_selected, train_y)
m_svm2.score(test_x_selected, test_y)   # 62.9


3) 반복적 선택(RFE)
 - 변수의 선택과 제거를 반복(통계 패키지의 step-wise기법과 유사)
 - 모든 변수를 학습 시킨 후 불필요한 변수를 제거, 다시 필요한 변수를 추가하는 방식
 
Y = x1 + x2 + x3 + ... + x10
Y =    + x2 + x3 + ... + x10 (x1 제거)
Y =    +    + x3 + ... + x10 (x2 제거)
Y = x1 +    + x3 + ... + x10 (x1 추가)

# iris data - feature selection(RFE)
from sklearn.feature_selection import RFE 

# 1. 데이터 로딩

# 2. RFE에 의한 변수 선택
m_fs3 = RFE(m_rf, n_features_to_select = 4)
m_fs3.fit(df_iris_new, df_iris.target)
m_fs3.get_support()                          # 선택된 변수 확인

m_fs3.ranking_                               # 변수 중요도 순서


# 그리드 서치
- 모델의 매개변수 선택 시 최적의 조합을 찾는 과정
- 매개변수들끼리 서로 독립적일 경우는 각각 베스트 매개변수를 구하면 됌
- 매개변수들끼리 서로 독립적이지 않을 경우는 
  특정 매개변수의 변화에 따른 다른 매개변수의 변화를 함께 고려
- train, validation, test 데이터 셋으로 진행

# 1. 데이터 로딩
df_iris = iris()

# 2. train/test 분리  
trainval_x, test_x, trainval_y, test_y = train_test_split(df_iris.data,
                                                          df_iris.target,
                                                          random_state=0)

# 3. 위 train data set을 train/validation 분리
train_x, val_x, train_y, val_y = train_test_split(trainval_x,
                                                  trainval_y,
                                                  random_state=0)

# 4. 모델 학습(위의 train set)  
m_knn = knn(n_neighbors=3)
m_knn.fit(train_x, train_y)

# 5. 매개변수 튜닝
best_score = 0

for i in np.arange(1,10) :
    m_knn = knn(n_neighbors=i)
    m_knn.fit(train_x, train_y)
    vscore = m_knn.score(val_x, val_y)
    
    if vscore > best_score :
        best_score = vscore
        best_params = i

best_score        
best_params

# 6. 매개변수 고정 후 모델 재학습
m_knn = knn(n_neighbors=3)
m_knn.fit(trainval_x, trainval_y)

# 7. 모델 평가
m_knn.score(test_x, test_y)     






