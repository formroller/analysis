# [ Y가 연속형인 경우의 분석 방법 ]
# 1. 전통 회귀분석
# - 여러가지 통계적 가정 필요
# - 가정이 만족되지 않으면 예측력이 떨어짐
# - 인과관계 파악 용이(특정 설명변수의 영향력)
# 
# 2. 분류분석 모델의 회귀분석 
# - 비통계적 모델로 통계적 가정 및 해석 필요 없음
# - 비교적 간단한 모델
# - 인과관계 파악 어려움

# 1. 분류 분석 
# [ 트리 기반 모델 ]
- DT > RF > GB > XGB
- outlier에 민감하지 않음
- 설명변수의 스케일 조정이 필요하지 않음
- 설명변수에 이진, 연속 모두 포함 가능

# [ 거리 기반 모델 ]
- knn
- outlier에 매우 민감
- 설명변수의 스케일 조정이 필요함
- 설명변수에 범주형이 많이 포함될수록 정확한 거리 측정 어려움
  
# 가장 가까운, 유사한 => 거리 측정

  소득  직업      지출  A상품구매여부
1 400  회사원(1)  200   X
2 410  회사원     220   X
3 500  자영업(2)  400   O

4 420  회사원     180   ?  (X일 가능성이 높아보임)


d14 <- sqrt((400-420)^2 + (1-1)^2 + (200-180)^2) # 28.28
d24 <- sqrt((410-420)^2 + (1-1)^2 + (220-180)^2) # 41.23
d34 <- sqrt((500-420)^2 + (2-1)^2 + (400-180)^2) # 234.23

결론 :
4번 관측치와 가장 가까운 이웃 1명을 고려시,
1번 관측치의 A상품구매여부를 따를것으로 예상


# [ 표준화가 중요한 이유 ]
#       x1    x2
# p1    5     100
# p2    6     200
# p3    10    150

# 1) 일반거리
d12 <- sqrt((5-6)^2 + (100-200)^2)
d13 <- sqrt((5-10)^2 + (100-150)^2)  
d12 > d13

# 2) 표준회된 거리
# (x1의 평균 6, 표준편차 0.01)
# (x2의 평균 150, 표준편차 30)

# 표준화 = (x - 평균)/표준편차
#       x1    x2
# p1    -100  -1.67
# p2    0     1.67
# p3    400   0

scale(iris$Sepal.Length)


# knn 모델
- k개의 가장 가까운(거리기반) 이웃을 찾아
  이웃이 갖는 범주로 예측을 수행하는 모델
- 적절한 k는 튜닝될 필요 있음
- Y의 종류가 2개일 경우, k가 짝수일 경우 동률발생으로 인해
  예측력이 떨어질 수 있음
  예) k=4, 2명이 A, 2명이 B => "A"

installed.packages('class')
library(class)

knn(train_x,  # 훈련 X set
    test_x,   # 예측 X set
    train_y,  # 훈련 Y set
    k = 3,    # 이웃의 수
    prob=T)   # 확률 출력 여부

# 1. sampling
v_rn <- sample(1:nrow(iris), size=0.7*nrow(iris))
iris_tr_x <- iris[v_rn, -5]  # Species 컬럼 제외
iris_tr_y <- iris[v_rn,  5]  # Species 컬럼 선택
iris_te_x <- iris[-v_rn, -5] # Species 컬럼 제외
iris_te_y <- iris[-v_rn,  5] # Species 컬럼 선택


# 2. 모델 학습
m_knn <- knn(iris_tr_x, iris_te_x, iris_tr_y, k=3, prob = T)

# 3. 모델 평가
sum(m_knn == iris_te_y) / nrow(iris_te_x) * 100

# 4. 새로운 데이터에 대한 예측
newdata <- iris_te_x[7, ]
knn(iris_tr_x, newdata, iris_tr_y, k=3, prob = T)


# 5. 매개변수 튜닝
# k 변화에 따른 예측률 변화 추이

# [ cancer - knn]
cancer <- read.csv('cancer.csv')

# 1. sampling
rn <- sample(1:nrow(cancer), size = 0.7*nrow(cancer))

train_x <- cancer[rn, -c(1,2)]
train_y <- cancer[rn, 2]
test_x  <- cancer[-rn, -c(1,2)]
test_y  <- cancer[-rn, 2]

sample_n <- sample(1:nrow(test_x), size = 1)
newdata_x <- test_x[sample_n, ]
newdata_y <- test_y[sample_n]

test_x  <- test_x[-sample_n, ]
test_y  <- test_y[-sample_n]

# 2. model 생성
m_knn <- knn(train_x, test_x, train_y, k=3)

# 3. 평가
sum(m_knn==test_y) / nrow(test_x) * 100

# 4. k 튜닝
sc_tr <- c() ; sc_te <- c()
for (i in 1:10) {
  m_knn1 <- knn(train_x, train_x, train_y, k=i)
  m_knn2 <- knn(train_x, test_x, train_y, k=i)
  
  sc_tr <- c(sc_tr, sum(m_knn1==train_y)/nrow(train_x) * 100)
  sc_te <- c(sc_te, sum(m_knn2==test_y)/ nrow(test_x) * 100)
}

dev.new()
plot(1:10, sc_tr, type = 'o', col = 'blue', ylim=c(80,100),
     xlab='k수', ylab='score', main='k에 따른 예측률 변화')
lines(1:10, sc_te, type = 'o', col = 'red')
legend(8,100, c('train','test'),col = c('blue','red'), lty=1)


# [ 참고 - 5번 교차검증 ]
# 훈련셋 5개  -> 5개의 train score의 평균
# 검정셋 5개  -> 5개의 test score의 평균
# 
# CV기법의 교차검증과는 데이터셋을 나누는 기준이 조금 다름
# 






# 데이터 분석 분류
# 1. 지도학습
#  1) 회귀분석
#  2) 분류분석
#    - 트리기반 : DT, RF, GB, XGB
#    - 거리기반 : knn
# 
# 2. 비지도학습
#  1) 군집분석 : kclust, k-means(거리기반)
#  2) 연관성분석
 
# [ 군집 분석 ]
# - 대표적인 비지도학습 모델
# - 데이터들의 유사성 기반으로 세분화, 데이터 축소 테크닉
# - 거리모델의 특징 모두 갖음
# - 이상치 민감, 스케일 조정 필요
# - 선택되어진 변수의 조합이 중요한 모델











