# [ ���� �м� ]
# - Y�� ���� �������н�
# - �������� ���з��� �� ������ Ư���� �ľ��ϴ� ���̴� �۾�
# - ������ ��� ��ũ��(clustering)
# - �з��� ������ Ư���� �ľ��� ���� �����н����ε� �߰� ���� ����
# - �Ÿ���� ��

# 1. ������ �����м�
# - �Ÿ��� ���� ª�� ������ ����Ʈ��� ���� �α��� ����������Ʈ�� 
#   ���Խ�Ű�� �������� ���� ����(������)
# - Ư�� ������ �ѹ� ���ԵǸ� ������ ���� �Ұ�
# - ������ �����ϴ� ������ ���� �������� ��� ����

# ** ���� ���� ���� : ����������Ʈ�� �������� �Ÿ� ���� ���
# 1. �ִܰŸ���(single, min)   : �Ÿ��� �ּҰ��� �������� �Ÿ��� ���
# 2. ����Ÿ���(complete, max) : �ִ밪�� ����
# 3. ��հŸ���(average)       : ��հ��� ����
# 4. �߾ӰŸ���(median)        : �߾Ӱ��� ����

# [ ���� - ���� ��������(min) ]
v1 <- c(1,3,6,10,18)
names(v1) <- paste('p',1:5,sep='')

dist(v1)
#     p1 p2 p3 p4
# p2  2         
# p3  5  3      
# p4  9  7  4   
# p5 17 15 12  8
# 
# step1) �� ����ġ������ �Ÿ� ���, ���� ����� �� ����ġ ��������
#        => C1(p1,p2)
# 
# step2) C1(p1,p2)�� ������ ����ġ���� �Ÿ� ���
# d(C1,p3) = min(d(p1,p3), d(p2,p3)) = min(5,3) = 3
# d(C1,p4) = min(d(p1,p4), d(p2,p4)) = min(9,7) = 7
# d(C1,p5) = min(d(p1,p5), d(p2,p5)) = min(17,15) = 15
# 
#     C1 p3 p4
# p3  3  .  4      
# p4  7  4  . 
# p5  15 12 8
# 
# step3) �� �Ÿ��� ���� ª���Ÿ� ���, ���ο� ���� ���� �Ǵ�
#        ���� ������ ���ο� ����������Ʈ �߰�
#        => C1(p1,p2,p3)
#        
# step4) C1(p1,p2,p3)�� ������ ����ġ���� �Ÿ� ���       
# d(C1,p4) = min(d(p1,p4), d(p2,p4), d(p3,p4)) = min(9,7,4) = 4
# d(C1,p5) = min(d(p1,p5), d(p2,p5), d(p3,p5)) = min(17,15,12) = 12
# 
#     C1 p4
# p4  4   . 
# p5  12  8   

# step5) �� �Ÿ��� ���� ª���Ÿ� ���, ���ο� ���� ���� �Ǵ�
# ���� ������ ���ο� ����������Ʈ �߰�
#        => C1(p1,p2,p3,p4)


# [ ���� - ���� ��������(min) �ڵ� ���� ]
d1 <- dist(v1)
hclust(d,                     # �Ÿ���� 
       method = ('complete',  # ������ ���� ����ġ���� �Ÿ� ����
                 'single',
                 'average', 
                 'median'))

m_clust <- hclust(d1, method = 'single')

# ���� �ð�ȭ
dev.new()
plot(m_clust, 
     hang = -1,  # ����Ʈ���� ��ġ�� x�࿡ �����ϱ� ���� �ɼ� 
     cex = 0.8, 
     main = 'single ����� ���� ������ ������������')

rect.hclust(tree = m_clust,  # clust ��
            k = 2)           # k(cluster)�� ��


# [ ���� ���� - iris�� ���������� ������ �����м� ���� ]
# 1) �Ÿ� Ȯ��

iris[c(1,2),]
#     Sepal.Length Sepal.Width Petal.Length Petal.Width Species
# 1          5.1         3.5          1.4         0.2  setosa
# 2          4.9         3.0          1.4         0.2  setosa

# d(p1,p2) = sqrt((5.1-4.9)^2 + (3.5-3.0)^2 + 0 + 0)
  
# 2) �� ����
d1 <- dist(iris[,-5])

# 2-1) single
m1 <- hclust(d1, method = 'single')

# 2-2) complete
m2 <- hclust(d1, method = 'complete')

# 2-3) average
m3 <- hclust(d1, method = 'average')

# 3) �ð�ȭ
dev.new()
par(mfrow=c(1,3))
plot(m1, hang = -1, cex = 0.7, main='single')
rect.hclust(m1, k=3)

plot(m2, hang = -1, cex = 0.7, main='complete')
rect.hclust(m2, k=3)

plot(m3, hang = -1, cex = 0.7, main='average')
rect.hclust(m3, k=3)

# 4) ��
# iris �������� Y���� �˰������Ƿ� �����м� ����� ��, ��ġ�� Ȯ��

# 1) single
# cutree�Լ��� �� ������ ��ȣ �Ҵ�
c1 <- cutree(m1,    # �����м� ��
             k = 3) # ������

y_result <- iris$Species
levels(y_result) <- c(1,2,3)

sum(y_result == c1) / nrow(iris) * 100  # 68

# 2) complete
c2 <- cutree(m2, k = 3)
sum(y_result == c2) / nrow(iris) * 100  # 49

# 3) average
c3 <- cutree(m3, k = 3)
sum(y_result == c3) / nrow(iris) * 100  # 90


# [ ���� ���� - iris �����͸� Ʃ���Ͽ� �����м� ���� ]
# 1) ��������
# �̻�ġ Ȯ��
dev.new()
plot(iris[,-5], col=iris$Species)

# ���� ǥ��ȭ
iris_sc <- scale(iris[,-5])
iris_sc2 <- scale(iris[,c(3,4)])

# 2) �𵨻���
d1 <- dist(iris_sc)
d2 <- dist(iris_sc2)

# 4�� ���������� ���� ��
m1 <- hclust(d1, 'single')
m2 <- hclust(d1, 'complete')
m3 <- hclust(d1, 'average')

# �ֿ� 2�� ���������� ���� ��
m11 <- hclust(d2, 'single')
m22 <- hclust(d2, 'complete')
m33 <- hclust(d2, 'average')

# 3) ����
# 4�������� ���� �� ��
c1 <- cutree(m1,3)
c2 <- cutree(m2,3)
c3 <- cutree(m3,3)

sum(y_result == c1) / nrow(iris) * 100  # single ����(66)
sum(y_result == c2) / nrow(iris) * 100  # compelte ����(78.67)
sum(y_result == c3) / nrow(iris) * 100  # average ����(68.67)

# 2�������� ���� �� ��
c11 <- cutree(m11,3)
c22 <- cutree(m22,3)
c33 <- cutree(m33,3)

sum(y_result == c11) / nrow(iris) * 100  # single ����(67.33)
sum(y_result == c22) / nrow(iris) * 100  # compelte ����(50)
sum(y_result == c33) / nrow(iris) * 100  # average ����(98)

# 4) �ð�ȭ
dev.new()
plot(m33, hang = -1, cex = 0.8,
     main = 'average�� ���� �����м� ���')
rect.hclust(m33,k=3)

# [ ���� - ������ ������ �� ���ϱ� ]
# ������ ���� �̹� ������ �ְų�, �����м��� ����� ���� ���ϴµ�
# ������ ������ ���� ������ �� �ִ� ��ǥ ����

install.packages('NbClust')
library(NbClust)

iris_sc1 <- scale(iris[,-5])
iris_sc2 <- scale(iris[,c(3,4)])

dev.new()
nc1 <- NbClust(data = iris_sc2 ,  # ������(�Ÿ���� �ƴ�)
               distance = 'euclidean',       # �Ÿ������
               min.nc = 2,                   # �ּ� ������
               max.nc = 10,                  # �ִ� ������
               method = 'average')           # �������� �Ÿ� ��� ���

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

# 2. ������� �����м�
# - �̹� �� ������ ���ԵǾ����� �ٸ� �������� �̵� ����
# - ��� ������ �߽��� �̵�, �Ÿ��� �����Ͽ� ������ �ٽ� ������
#   ������ �ݺ�, ���̻� ������ �߽��� ��ȭ�� ������ stop
# - �� metric ����(������ �л�, ������ �л�)

# ������� �����м��� ������������
# 1. �ʱⰪ ����(����)
# 2. �����ϰ� ������ ����ġ�κ��� ���� ����� ����ġ�� ���� ����
# 3. ���� �Ÿ� ���, ���ο� Ŭ������ ������ ������ ����� ����
# 4. �̵��� ������ �߽ɱ��� ���� ����ġ���� �Ÿ��� ��� ���� 
#    (������ ���ԵǾ� �ִ� ����ġ�鵵 ��� ��������ġ�� ���� ���)
# 5. ���� Ư�� ����ġ���� �Ÿ��� ���� �������� ���� �̵��� ������ 
#    �� �����ٸ� ����ġ�� ������ �̵�
# 6. ���̻� ��� ������ �߽��� ��ȭ�� ���������� ���� �ݺ�

# ���� �� �� �ʿ��� �л�
# - �Ѻл�(total_ss) : within_ss + between_ss
# - �������л�(within_ss) : Ư�� ���� �������� �л�
# - �������л�(between_ss) : �� �������� �л�


# [ iris �����͸� ����� k-means �����м� ]
# 1. �� ����
m_kmean <- kmeans(iris[,-5],   # ������(�Ÿ���� �ƴ�)
                  3)           # k�Ǽ�

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
# (between_SS / total_SS =  88.4 %)  # ������
# 
# Available components:
#   
# [1] "cluster"      "centers"      "totss"        "withinss"    
# [5] "tot.withinss" "betweenss"    "size"         "iter"        
# [9] "ifault"

# 2. �� ��
m_kmean$totss     # �Ѻл�(681.3706 = 78.85 + 602.5192)
m_kmean$withinss  # �׷쳻 �л�(78.85)
                  # sum(m_kmean$withinss)
m_kmean$betweenss # �׷찣 �л�(602.5192)
    
# 1) �л����� �ؼ�(������)
m_kmean$betweenss / m_kmean$totss * 100   # 88.42753

m_kmean$cluster   

# 2) ��ġ���� �ؼ�
y_result2 <- iris$Species
levels(y_result2) <- c(2,1,3)

sum(y_result2 == m_kmean$cluster) / nrow(iris) * 100

# 3. ������ K�� ã��
vscore <- c()

for (i in 1:10) {
  m_kmean <- kmeans(iris[,-5], i)
  vscore <- c(vscore, m_kmean$betweenss / m_kmean$totss * 100)
}

dev.new()
plot(1:10, vscore, type='o', col='blue',
     xlab = '������(k)', ylab = 'betweenss / totss',
     main='�������� ��ȭ�� ���� betweenss / totss')

    

    










