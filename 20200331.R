# [ 트리 기반 모델의 변수 선택 기능 ]
트리기반 모델은 모델 내 변수의 중요도를 파악,
해당 변수 중 중요한 변수들을 선택하여 배치하는 기능을 가짐
=> 다른 모델에 비해 변수 선택에 대한 부담이 없음
=> final model로 사용하지 않더라도 변수 연구 시 사용
(모델 기반 변수 선택 방식)

# 불순도
하위 노드에 서로 다른 class로 구성되어 있는 정도를 나타낸 수치
대표적인 불순도 측정 지수 => "지니 불순도"

# 지니 불순도 측정 방식(Y가 2개의 class로 구성되어 있는 경우)
f(p) = p(1-p), p는 특정 class가 속할 확률의미

ex)  (OOXXX)로 구성된 노드의 지니불순도 계산
p=O가 선택될 확률이라 가정, p=2/5, 1-p=3/5
f(p) = 2/5 * 3/5 = 0.24


# 트리기반 모델 내부의 변수 중요도 확인
m_dt$variable.importance

Petal.Width Petal.Length Sepal.Length  Sepal.Width 
61.20231     58.32399     35.90656     22.37384

30/5 => 97%
30   => 98%

# 조건부 추론 나무(ctree)
# 기존 의사결정나무가 각 노드의 선택에 통계적 유의성 검정을
# 할 수 없다는 단점을 보완

install.packages('party')
library(party)

m1 <- ctree(formula = Species ~ . , 
            data=iris)
m1          # 모델 확인

dev.new()
plot(m1)

# 분석 시 기타 고려사항
# 1. Sampling 할때마다 다른 모델이 생성, 평가점수 달라지는 현상
# => cross validation(CV, 교차검증)

# 2. 변수 조합, 변형 고려
# X1X2, (X1 + X2)/2, log(x1)


# 2) Random Forest   
# - decision tree의 단일의사결정 방식을 보완, 다수의 tree 구성
# - 여러개의 tree를 구성하여 종합 결론을 내는 방식
# - randomForest-regressor, randomForest-classfier 존재
# - 평균 또는 투표에 의해 최종 결론
# - 서로 다른 트리를 구성

install.packages('randomForest')
library(randomForest)

# 1) sampling
v_rn <- sample(1:nrow(iris), size = nrow(iris) * 0.7)
tr_iris <- iris[v_rn, ]
te_iris <- iris[-v_rn, ]

# 2) training
m_rf <- randomForest(Species ~ ., data=tr_iris)

# 특성 중요도 확인
m_rf$importance

#                MeanDecreaseGini
# Sepal.Length         6.637587
# Sepal.Width          1.122958
# Petal.Length        35.381573
# Petal.Width         26.070911

# 3) score
v_predict <- predict(m_rf, newdata = te_iris, type = 'class')
sum(te_iris$Species == v_predict) / nrow(te_iris) * 100

# 4) predict(확보된 데이터의 일부 추출 후 예측 수행)
v_rn2 <- sample(1:nrow(te_iris), size = 1)
v_newdata <- te_iris[v_rn2, ]   

predict(m_rf, newdata = v_newdata)   # 예측값
v_newdata$Species                    # 실제값

# 5) 매개변수 튜닝
# 5-1) ntree : 트리의 개수
randomForest(Species ~ ., data=tr_iris, ntree=5)

# elbow point
v_score <- c()
for (i in 1:1000) {
  m_rf <- randomForest(Species ~ ., data=tr_iris, ntree=i)
  v_pr <- predict(m_rf, newdata = te_iris, type = 'class')
  vscore <- sum(te_iris$Species == v_pr) / nrow(te_iris) * 100
  v_score <- c(v_score, vscore)
}

dev.new()
plot(1:1000, v_score, type='o', 
     xlab = 'ntree', ylab = 'score', 
     main='RF에서의 ntree변화에 따른 예측률 변화')


# 5-2) mtry 
# - 각 노드에 설명변수 선택시 고려되는 후보의 개수
# - 서로 다른 트리를 구성하기위해 만든 매개변수
# - 클수록 많은 설명변수를 고려, 서로 비슷한 트리 구성,
# - 작을수록 전체 중 랜덤하게 선택된 일부 설명변수만 고려,
#   서로 다른 트리 구성, 각 트리가 매우 복잡해질 가능성 

# [ 연습 문제 ]
# mtry 값의 변화에 따른 예측률 및 과대적합의 변화 확인
v_score_tr <- c() ; v_score_te <- c()

for (i in 1:4) {
  # 모델 생성
  m_rf <- randomForest(Species ~ ., data=tr_iris, mtry=i)
  # train set score
  v_pr_tr <- predict(m_rf, newdata = tr_iris, type = 'class')
  vscore_tr <- sum(tr_iris$Species==v_pr_tr) / nrow(tr_iris) * 100
  # test set score
  v_pr_te <- predict(m_rf, newdata = te_iris, type = 'class')
  vscore_te <- sum(te_iris$Species==v_pr_te) / nrow(te_iris) * 100
  
  v_score_tr <- c(v_score_tr, vscore_tr)
  v_score_te <- c(v_score_te, vscore_te)
}

dev.new()
plot(1:4, v_score_te, type='o', col='red', ylim=c(80,100),
     xlab = 'mtry', ylab = 'score', 
     main='RF에서의 mtry변화에 따른 예측률 변화')
lines(1:4, v_score_tr, type='o', col='blue')


# [ 참고 - random forest의 회귀 ]
data("airquality")
airquality

str(airquality)  # Ozone : 오존량(Y), 나머지 설명변수 가정
m_rf_reg <- randomForest(Ozone ~ . , 
                         data = airquality,
                         mtry = 3,
                         na.action = na.omit)
m_rf_reg



mean((12000 - 15000)^2 , (13000 - 13100)^2)
mean((12 - 15)^2 , (13 - 13)^2)


# Tree 기반 모델
Decision Tree(DT) -> Random Forest(RF)
-> Gradiant Boosting Tree(GB)
-> eXtreme Gradiant Boosting Tree(XGB)

rpart(Y ~ X, data=)        # Y는 non-factor여도 가능
randomForest(Y ~ X, data=) # Y는 factor여야 가능

# 거리 기반 모델
# - 각 관측치의 거리를 계산, 가장 가까운 데이터의 행동을 그대로 추천
# - 추천시스템, 이미지 인식 시스템의 기본이 되는 알고리즘
# - 거리는 스케일 조정이 반드시 필요
# - 이상치에 매우 민감한 모델
 






