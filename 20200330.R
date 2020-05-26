# [ 데이터 분석 분류 ]

# - 머신러닝(기계학습) : 
#   사용자가 직접 데이터의 패턴, rule을 찾지 않고
#   데이터의 학습을 통해 자동으로 패턴을 찾을수 있도록 하는 분석기법

# 1. 지도학습(supervised learning)     : Y 존재
#    X ~ Y
#    => Y에 영향을 미치는 설명변수의 최적의 조합을 찾는 과정
#       (변수의 선택 및 제거, 변수의 결합 및 변형...)
   
# c1 = (x1 + x2)/2
# c2   = log(x1)
# ....

# 1) 회귀기반 모델(Y가 연속형) : 부동산 가격 예측, 수요예측...
# 2) 분류기반 모델(Y가 범주형) : 생존여부 예측, 이탈여부 예측 ...


# 2. 비지도학습(unsupervised learning) : Y 존재 X
#    X ~ => X들이 갖는 각 관측치의 유사성 기반 데이터의 세분화

# - 딥러닝 : 
#   머신러닝의 일부, 딥러닝에 대한 연구가 강화되면서
#   기존 머신러닝에서 분리하여 설명하는 추세
#   주로 인공신경망 모델을 사용한 분석기법을 의미
#   머신러닝에 비해 비정형화된 데이터 처리에 용이함
   
   
# [ 분류 분석 - 지도학습(Y가 범주형)]
#    X들이 가지는 패턴에 따라 Y의 종류를 예측하기 위한 분석
   
# 1. 분류분석 과정  
#    - 데이터 수집(Y에 영향을 미칠만한 모든 데이터)
#    - 변수연구(feature selection/변수결합/변수변형)
#    - 이상치 및 결측치 처리
#    - 모델 선택
#    - data sampling(train set/(validation set)/test set)
#    - 모델 학습(train set)
#    - 모델 평가(test set)
#      실제 정답과 예측값과의 일치율로 평가
#    - 모델 튜닝(validation set)
#    - 결과 해석
   

# Y(이탈/비이탈) = X1 + 2X2 + 3X3 + ... + X100   
   
   #                   Y        X
   # 수집 data  :     Y 존재    x1 x2 ... x100
   #     학습용 :     Y 존재    x1 x2 ... x100
   #     평가용 :     Y 존재    x1 x2 ... x100
   #                     
   # new data   :     Y 모름    x1 x2 ... x100
   

# 1) Decision Tree(의사결정나무)
#   - 분류분석을 수행하는 트리기반 모델의 가장 초기 모델
#   - 한번의 패턴확인으로 Y를 예측
#   - 패턴이 Tree 구조를 띔
#   - 비통계적 모델로 모델 적용 및 해석이 매우 쉽다
#   - 단일의사결정이므로 예측력이 불안하거나 overfit될수 있음

# [ iris data set - DT ]
install.packages('rpart')
library(rpart)

# 1. sampling
v_rn <- sample(1:nrow(iris), size = nrow(iris) * 0.7)
iris_train <- iris[v_rn, ]
iris_test  <- iris[-v_rn, ]

# 2. 모델 생성(training, pattern 추출)
m_dt <- rpart(data=iris_train, formula = Species ~ .)

# 모델 확인(각 노드의 분류율 및 설명변수의 선택과정)
m_dt

# 3. 모델 시각화
dev.new()
plot(m_dt, compress = T)
text(m_dt, cex=1.5)

install.packages('rpart.plot')
library(rpart.plot)

prp(m_dt, type=4, extra=2, digits = 3)

# 4. 모델 평가
v_y <- iris_test$Species
v_predict <- predict(m_dt, newdata=iris_test, type = 'class')

sum(v_y == v_predict) / nrow(iris_test) * 100

# 모델을 통한 새로운 데이터에 대한 Y값 예측 
iris_new = data.frame(Sepal.Length=4.7, 
                      Sepal.Width=3.8, 
                      Petal.Length=1.9, 
                      Petal.Width=0.3)
# 5. 모델 튜닝
# 1) 매개변수 값 확인
m_dt$control

# 2) 매개변수 종류
# 1-1) minbucket
# - 추가 하위 노드를 생성하기 위한 오분류 데이터의 최소 기준
# - 값이 7인 경우 오분류 데이터가 7미만일 경우는 더이상 분류하지 X
# - 값이 작을수록 모델의 복잡도는 커짐

# 1-2) minsplit
# - minbucket = round(minsplit/3)

# 1-3) cp
# 1-4) maxdepth

# 3) 매개변수 조정 
m_dt2 <- rpart(data=iris_train, 
               formula = Species ~ . ,
               control = rpart.control(minbucket=2))

m_dt2

# 3) 최적의 매개변수 값 확인
# [ 연습 문제 ]
# minbucket이 1~10 변화될때의 예측력 변화 확인
# overfit 여부  
v_tr_score <- c() ; v_te_score <- c()

for (i in 1:10) {
  m_dt <- rpart(data=iris_train, 
                formula = Species ~ . ,
                control = rpart.control(minbucket=i))

  # train data score
  tr_pre <- predict(m_dt, newdata = iris_train, type = 'class')
  tr_score <- sum(iris_train$Species==tr_pre)/nrow(iris_train) * 100
  
  # test data score
  te_pre <- predict(m_dt, newdata = iris_test, type = 'class')
  te_score <- sum(iris_test$Species==te_pre)/nrow(iris_test) * 100
  
  # 벡터화
  v_tr_score <- c(v_tr_score, tr_score)
  v_te_score <- c(v_te_score, te_score)
}

# 시각화
dev.new()
plot(1:10, v_tr_score, type = 'o', col='blue', ylim = c(90,100))
lines(1:10, v_te_score, type = 'o', col='red')


# ** overfit(과대적합) : train data set에 대한 예측을 과도하게 
# 강하게 함으로 인해 모델이 복잡해져 새로운 데이터(test data)에 대한
# 예측력과의 차이가 꽤 발생하는 경우를 의미


# 6. 새로운 데이터에 대한 예측
predict(m_dt, newdata=iris_new, type = 'class')

# [ 연습 문제 ]
# cancer.csv 파일은 환자들의 암 종양의 데이터를 수집,
# 이를 통해 양성/악성을 분류할 수 있는 모델을 생성(decision tree)
cancer <- read.csv('cancer.csv')
str(cancer)

# 교차 산점도를 통한 각 설명변수의 의미 파악
dev.new()
plot(cancer[, 3:11], col = cancer$diagnosis)

# 2) Random Forest   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

     