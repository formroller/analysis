# [ 군집 분석 ]
# - Y가 없는 비지도학습
# - 모집단을 세분류한 후 집단의 특성을 파악하는 마이닝 작업
# - 데이터 축소 테크닉(clustering)
# - 분류된 집단의 특성을 파악한 이후 지도학습으로도 추가 연구 가능
# - 거리기반 모델

# 1. 계층적 군집분석
# - 거리가 가장 짧은 데이터 포인트들로 부터 인근으 데이터포인트를 
#   포함시키는 방향으로 군집 형성(순차적)
# - 특정 군집에 한번 포함되면 군집은 수정 불가
# - 군집을 형성하는 과정에 따라 여러가지 방식 제공

# ** 군집 형성 과정 : 데이터포인트와 군집과의 거리 측정 방식
# 1. 최단거리법(single, min)   : 거리중 최소값을 군집과의 거리로 결론
# 2. 최장거리법(complete, max) : 최대값을 선택
# 3. 평균거리법(average)       : 평균값을 선택
# 4. 중앙거리법(median)        : 중앙값을 선택

# [ 예제 - 군집 형성과정(min) ]
v1 <- c(1,3,6,10,18)
names(v1) <- paste('p',1:5,sep='')

dist(v1)
#     p1 p2 p3 p4
# p2  2         
# p3  5  3      
# p4  9  7  4   
# p5 17 15 12  8
# 
# step1) 각 관측치끼리의 거리 계산, 가장 가까운 두 관측치 군집형성
#        => C1(p1,p2)
# 
# step2) C1(p1,p2)과 나머지 관측치와의 거리 계산
# d(C1,p3) = min(d(p1,p3), d(p2,p3)) = min(5,3) = 3
# d(C1,p4) = min(d(p1,p4), d(p2,p4)) = min(9,7) = 7
# d(C1,p5) = min(d(p1,p5), d(p2,p5)) = min(17,15) = 15
# 
#     C1 p3 p4
# p3  3  .  4      
# p4  7  4  . 
# p5  15 12 8
# 
# step3) 위 거리중 가장 짧은거리 기반, 새로운 군집 형성 또는
#        기존 군집에 새로운 데이토포인트 추가
#        => C1(p1,p2,p3)
#        
# step4) C1(p1,p2,p3)과 나머지 관측치와의 거리 계산       
# d(C1,p4) = min(d(p1,p4), d(p2,p4), d(p3,p4)) = min(9,7,4) = 4
# d(C1,p5) = min(d(p1,p5), d(p2,p5), d(p3,p5)) = min(17,15,12) = 12
# 
#     C1 p4
# p4  4   . 
# p5  12  8   

# step5) 위 거리중 가장 짧은거리 기반, 새로운 군집 형성 또는
# 기존 군집에 새로운 데이토포인트 추가
#        => C1(p1,p2,p3,p4)


# [ 예제 - 군집 형성과정(min) 코드 구현 ]
d1 <- dist(v1)
hclust(d,                     # 거리행렬 
       method = ('complete',  # 군집과 개별 관측치와의 거리 기준
                 'single',
                 'average', 
                 'median'))

m_clust <- hclust(d1, method = 'single')

# 모델의 시각화
dev.new()
plot(m_clust, 
     hang = -1,  # 포인트들의 위치를 x축에 고정하기 위한 옵션 
     cex = 0.8, 
     main = 'single 기법에 의한 계층적 군집형성과정')

rect.hclust(tree = m_clust,  # clust 모델
            k = 2)           # k(cluster)의 수


# [ 연습 문제 - iris의 설명변수만 가지고 군집분석 수행 ]
# 1) 거리 확인

iris[c(1,2),]
#     Sepal.Length Sepal.Width Petal.Length Petal.Width Species
# 1          5.1         3.5          1.4         0.2  setosa
# 2          4.9         3.0          1.4         0.2  setosa

# d(p1,p2) = sqrt((5.1-4.9)^2 + (3.5-3.0)^2 + 0 + 0)
  
# 2) 모델 생성
d1 <- dist(iris[,-5])

# 2-1) single
m1 <- hclust(d1, method = 'single')

# 2-2) complete
m2 <- hclust(d1, method = 'complete')

# 2-3) average
m3 <- hclust(d1, method = 'average')

# 3) 시각화
dev.new()
par(mfrow=c(1,3))
plot(m1, hang = -1, cex = 0.7, main='single')
rect.hclust(m1, k=3)

plot(m2, hang = -1, cex = 0.7, main='complete')
rect.hclust(m2, k=3)

plot(m3, hang = -1, cex = 0.7, main='average')
rect.hclust(m3, k=3)

# 4) 평가
# iris 데이터의 Y값을 알고있으므로 군집분석 결과와 비교, 일치율 확인

# 1) single
# cutree함수로 각 군집의 번호 할당
c1 <- cutree(m1,    # 군집분석 모델
             k = 3) # 군집수

y_result <- iris$Species
levels(y_result) <- c(1,2,3)

sum(y_result == c1) / nrow(iris) * 100  # 68

# 2) complete
c2 <- cutree(m2, k = 3)
sum(y_result == c2) / nrow(iris) * 100  # 49

# 3) average
c3 <- cutree(m3, k = 3)
sum(y_result == c3) / nrow(iris) * 100  # 90


# [ 연습 문제 - iris 데이터를 튜닝하여 군집분석 수행 ]
# 1) 변수가공
# 이상치 확인
dev.new()
plot(iris[,-5], col=iris$Species)

# 변수 표준화
iris_sc <- scale(iris[,-5])
iris_sc2 <- scale(iris[,c(3,4)])

# 2) 모델생성
d1 <- dist(iris_sc)
d2 <- dist(iris_sc2)

# 4개 설명변수에 대한 모델
m1 <- hclust(d1, 'single')
m2 <- hclust(d1, 'complete')
m3 <- hclust(d1, 'average')

# 주요 2개 설명변수에 대한 모델
m11 <- hclust(d2, 'single')
m22 <- hclust(d2, 'complete')
m33 <- hclust(d2, 'average')

# 3) 모델평가
# 4개변수에 대한 모델 평가
c1 <- cutree(m1,3)
c2 <- cutree(m2,3)
c3 <- cutree(m3,3)

sum(y_result == c1) / nrow(iris) * 100  # single 점수(66)
sum(y_result == c2) / nrow(iris) * 100  # compelte 점수(78.67)
sum(y_result == c3) / nrow(iris) * 100  # average 점수(68.67)

# 2개변수에 대한 모델 평가
c11 <- cutree(m11,3)
c22 <- cutree(m22,3)
c33 <- cutree(m33,3)

sum(y_result == c11) / nrow(iris) * 100  # single 점수(67.33)
sum(y_result == c22) / nrow(iris) * 100  # compelte 점수(50)
sum(y_result == c33) / nrow(iris) * 100  # average 점수(98)

# 4) 시각화
dev.new()
plot(m33, hang = -1, cex = 0.8,
     main = 'average에 의한 군집분석 결과')
rect.hclust(m33,k=3)

# [ 참고 - 적절한 군집의 수 구하기 ]
# 군집의 수는 이미 정해져 있거나, 군집분석의 결과를 보고 정하는데
# 적절한 군집의 수를 참고할 수 있는 지표 제공

install.packages('NbClust')
library(NbClust)

iris_sc1 <- scale(iris[,-5])
iris_sc2 <- scale(iris[,c(3,4)])

dev.new()
nc1 <- NbClust(data = iris_sc2 ,  # 데이터(거리행렬 아님)
               distance = 'euclidean',       # 거리계산방법
               min.nc = 2,                   # 최소 군집수
               max.nc = 10,                  # 최대 군집수
               method = 'average')           # 군집과의 거리 계산 방법

# ******************************************************************* 
# * Among all indices:                                                
# * 8 proposed 2 as the best number of clusters 
# * 9 proposed 3 as the best number of clusters 
# * 1 proposed 4 as the best number of clusters 
# * 1 proposed 7 as the best number of clusters 
# * 2 proposed 8 as the best number of clusters 
# * 1 proposed 9 as the best number of clusters 
# * 1 proposed 10 as the best number of clusters 
# 
# ***** Conclusion *****                            
# 
# * According to the majority rule, the best number of clusters is  3 

# 2. 비계층적 군집분석
# - 이미 한 군집에 포함되었더라도 다른 군집으로 이동 가능
# - 계속 군집의 중심을 이동, 거리를 재계산하여 군집을 다시 형성을
#   수차례 반복, 더이상 군집의 중심의 변화가 없을때 stop
# - 평가 metric 존재(군집간 분산, 군집내 분산)

# 비계층적 군집분석의 군집형성과정
# 1. 초기값 생성(랜덤)
# 2. 랜덤하게 생성된 관측치로부터 가장 가까운 관측치와 군집 형성
# 3. 이후 거리 계산, 새로운 클러스터 형성은 계층적 방법과 동일
# 4. 이동된 군집의 중심기준 기존 관측치와의 거리를 모두 재계산 
#    (군집내 포함되어 있는 관측치들도 모두 개별관측치로 보고 계산)
# 5. 만약 특정 관측치와의 거리가 기존 군집보다 새로 이동된 군집과 
#    더 가깝다면 관측치의 군집이 이동
# 6. 더이상 모든 군집의 중심의 변화가 없을때까지 무한 반복

# 군집 평가 시 필요한 분산
# - 총분산(total_ss) : within_ss + between_ss
# - 군집내분산(within_ss) : 특정 군집 내에서의 분산
# - 군집간분산(between_ss) : 각 군집간의 분산


# [ iris 데이터를 사용한 k-means 군집분석 ]
# 1. 모델 생성
m_kmean <- kmeans(iris[,-5],   # 데이터(거리행렬 아님)
                  3)           # k의수

m_kmean

# K-means clustering with 3 clusters of sizes 62, 50, 38
# 
# Cluster means:
#   Sepal.Length Sepal.Width Petal.Length Petal.Width
# 1     5.901613    2.748387     4.393548    1.433871
# 2     5.006000    3.428000     1.462000    0.246000
# 3     6.850000    3.073684     5.742105    2.071053
# 
# Clustering vector:
# [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# [34] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1
# [67] 1 1 1 1 1 1 1 1 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# [100] 1 3 1 3 3 3 3 1 3 3 3 3 3 3 1 1 3 3 3 3 1 3 1 3 1 3 3 1 1 3 3 3 3
# [133] 3 1 3 3 3 3 1 3 3 3 1 3 3 3 1 3 3 1
# 
# Within cluster sum of squares by cluster:
#   [1] 39.82097 15.15100 23.87947
# (between_SS / total_SS =  88.4 %)  # 평가점수
# 
# Available components:
#   
# [1] "cluster"      "centers"      "totss"        "withinss"    
# [5] "tot.withinss" "betweenss"    "size"         "iter"        
# [9] "ifault"

# 2. 모델 평가
m_kmean$totss     # 총분산(681.3706 = 78.85 + 602.5192)
m_kmean$withinss  # 그룹내 분산(78.85)
                  # sum(m_kmean$withinss)
m_kmean$betweenss # 그룹간 분산(602.5192)
    
# 1) 분산으로 해석(평가점수)
m_kmean$betweenss / m_kmean$totss * 100   # 88.42753

m_kmean$cluster   

# 2) 일치율로 해석
y_result2 <- iris$Species
levels(y_result2) <- c(2,1,3)

sum(y_result2 == m_kmean$cluster) / nrow(iris) * 100

# 3. 적절한 K수 찾기
vscore <- c()

for (i in 1:10) {
  m_kmean <- kmeans(iris[,-5], i)
  vscore <- c(vscore, m_kmean$betweenss / m_kmean$totss * 100)
}

dev.new()
plot(1:10, vscore, type='o', col='blue',
     xlab = '군집수(k)', ylab = 'betweenss / totss',
     main='군집수의 변화에 따른 betweenss / totss')

    

    











