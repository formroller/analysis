# [ Y�� �������� ����� �м� ��� ]
# 1. ���� ȸ�ͺм�
# - �������� ����� ���� �ʿ�
# - ������ �������� ������ �������� ������
# - �ΰ����� �ľ� ����(Ư�� ���������� �����)
# 
# 2. �з��м� ���� ȸ�ͺм� 
# - ������� �𵨷� ����� ���� �� �ؼ� �ʿ� ����
# - ���� ������ ��
# - �ΰ����� �ľ� �����

# 1. �з� �м� 
# [ Ʈ�� ��� �� ]
- DT > RF > GB > XGB
- outlier�� �ΰ����� ����
- ���������� ������ ������ �ʿ����� ����
- ���������� ����, ���� ��� ���� ����

# [ �Ÿ� ��� �� ]
- knn
- outlier�� �ſ� �ΰ�
- ���������� ������ ������ �ʿ���
- ���������� �������� ���� ���Եɼ��� ��Ȯ�� �Ÿ� ���� �����
  
# ���� �����, ������ => �Ÿ� ����

  �ҵ�  ����      ����  A��ǰ���ſ���
1 400  ȸ���(1)  200   X
2 410  ȸ���     220   X
3 500  �ڿ���(2)  400   O

4 420  ȸ���     180   ?  (X�� ���ɼ��� ���ƺ���)


d14 <- sqrt((400-420)^2 + (1-1)^2 + (200-180)^2) # 28.28
d24 <- sqrt((410-420)^2 + (1-1)^2 + (220-180)^2) # 41.23
d34 <- sqrt((500-420)^2 + (2-1)^2 + (400-180)^2) # 234.23

��� :
4�� ����ġ�� ���� ����� �̿� 1���� ������,
1�� ����ġ�� A��ǰ���ſ��θ� ���������� ����


# [ ǥ��ȭ�� �߿��� ���� ]
#       x1    x2
# p1    5     100
# p2    6     200
# p3    10    150

# 1) �ϹݰŸ�
d12 <- sqrt((5-6)^2 + (100-200)^2)
d13 <- sqrt((5-10)^2 + (100-150)^2)  
d12 > d13

# 2) ǥ��ȸ�� �Ÿ�
# (x1�� ��� 6, ǥ������ 0.01)
# (x2�� ��� 150, ǥ������ 30)

# ǥ��ȭ = (x - ���)/ǥ������
#       x1    x2
# p1    -100  -1.67
# p2    0     1.67
# p3    400   0

scale(iris$Sepal.Length)


# knn ��
- k���� ���� �����(�Ÿ����) �̿��� ã��
  �̿��� ���� ���ַ� ������ �����ϴ� ��
- ������ k�� Ʃ�׵� �ʿ� ����
- Y�� ������ 2���� ���, k�� ¦���� ��� �����߻����� ����
  �������� ������ �� ����
  ��) k=4, 2���� A, 2���� B => "A"

installed.packages('class')
library(class)

knn(train_x,  # �Ʒ� X set
    test_x,   # ���� X set
    train_y,  # �Ʒ� Y set
    k = 3,    # �̿��� ��
    prob=T)   # Ȯ�� ��� ����

# 1. sampling
v_rn <- sample(1:nrow(iris), size=0.7*nrow(iris))
iris_tr_x <- iris[v_rn, -5]  # Species �÷� ����
iris_tr_y <- iris[v_rn,  5]  # Species �÷� ����
iris_te_x <- iris[-v_rn, -5] # Species �÷� ����
iris_te_y <- iris[-v_rn,  5] # Species �÷� ����


# 2. �� �н�
m_knn <- knn(iris_tr_x, iris_te_x, iris_tr_y, k=3, prob = T)

# 3. �� ��
sum(m_knn == iris_te_y) / nrow(iris_te_x) * 100

# 4. ���ο� �����Ϳ� ���� ����
newdata <- iris_te_x[7, ]
knn(iris_tr_x, newdata, iris_tr_y, k=3, prob = T)


# 5. �Ű����� Ʃ��
# k ��ȭ�� ���� ������ ��ȭ ����

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

# 2. model ����
m_knn <- knn(train_x, test_x, train_y, k=3)

# 3. ��
sum(m_knn==test_y) / nrow(test_x) * 100

# 4. k Ʃ��
sc_tr <- c() ; sc_te <- c()
for (i in 1:10) {
  m_knn1 <- knn(train_x, train_x, train_y, k=i)
  m_knn2 <- knn(train_x, test_x, train_y, k=i)
  
  sc_tr <- c(sc_tr, sum(m_knn1==train_y)/nrow(train_x) * 100)
  sc_te <- c(sc_te, sum(m_knn2==test_y)/ nrow(test_x) * 100)
}

dev.new()
plot(1:10, sc_tr, type = 'o', col = 'blue', ylim=c(80,100),
     xlab='k��', ylab='score', main='k�� ���� ������ ��ȭ')
lines(1:10, sc_te, type = 'o', col = 'red')
legend(8,100, c('train','test'),col = c('blue','red'), lty=1)


# [ ���� - 5�� �������� ]
# �Ʒü� 5��  -> 5���� train score�� ���
# ������ 5��  -> 5���� test score�� ���
# 
# CV����� ������������ �����ͼ��� ������ ������ ���� �ٸ�
# 






# ������ �м� �з�
# 1. �����н�
#  1) ȸ�ͺм�
#  2) �з��м�
#    - Ʈ����� : DT, RF, GB, XGB
#    - �Ÿ���� : knn
# 
# 2. �������н�
#  1) �����м� : kclust, k-means(�Ÿ����)
#  2) �������м�
 
# [ ���� �м� ]
# - ��ǥ���� �������н� ��
# - �����͵��� ���缺 ������� ����ȭ, ������ ��� ��ũ��
# - �Ÿ����� Ư¡ ��� ����
# - �̻�ġ �ΰ�, ������ ���� �ʿ�
# - ���õǾ��� ������ ������ �߿��� ��










