# [ ������ �м� �з� ]

# - �ӽŷ���(����н�) : 
#   ����ڰ� ���� �������� ����, rule�� ã�� �ʰ�
#   �������� �н��� ���� �ڵ����� ������ ã���� �ֵ��� �ϴ� �м����

# 1. �����н�(supervised learning)     : Y ����
#    X ~ Y
#    => Y�� ������ ��ġ�� ���������� ������ ������ ã�� ����
#       (������ ���� �� ����, ������ ���� �� ����...)
   
# c1 = (x1 + x2)/2
# c2   = log(x1)
# ....

# 1) ȸ�ͱ�� ��(Y�� ������) : �ε��� ���� ����, ���俹��...
# 2) �з���� ��(Y�� ������) : �������� ����, ��Ż���� ���� ...


# 2. �������н�(unsupervised learning) : Y ���� X
#    X ~ => X���� ���� �� ����ġ�� ���缺 ��� �������� ����ȭ

# - ������ : 
#   �ӽŷ����� �Ϻ�, �����׿� ���� ������ ��ȭ�Ǹ鼭
#   ���� �ӽŷ��׿��� �и��Ͽ� �����ϴ� �߼�
#   �ַ� �ΰ��Ű�� ���� ����� �м������ �ǹ�
#   �ӽŷ��׿� ���� ������ȭ�� ������ ó���� ������
   
   
# [ �з� �м� - �����н�(Y�� ������)]
#    X���� ������ ���Ͽ� ���� Y�� ������ �����ϱ� ���� �м�
   
# 1. �з��м� ����  
#    - ������ ����(Y�� ������ ��ĥ���� ��� ������)
#    - ��������(feature selection/��������/��������)
#    - �̻�ġ �� ����ġ ó��
#    - �� ����
#    - data sampling(train set/(validation set)/test set)
#    - �� �н�(train set)
#    - �� ��(test set)
#      ���� ����� ���������� ��ġ���� ��
#    - �� Ʃ��(validation set)
#    - ��� �ؼ�
   

# Y(��Ż/����Ż) = X1 + 2X2 + 3X3 + ... + X100   
   
   #                   Y        X
   # ���� data  :     Y ����    x1 x2 ... x100
   #     �н��� :     Y ����    x1 x2 ... x100
   #     �򰡿� :     Y ����    x1 x2 ... x100
   #                     
   # new data   :     Y ��    x1 x2 ... x100
   

# 1) Decision Tree(�ǻ��������)
#   - �з��м��� �����ϴ� Ʈ����� ���� ���� �ʱ� ��
#   - �ѹ��� ����Ȯ������ Y�� ����
#   - ������ Tree ������ ��
#   - ������� �𵨷� �� ���� �� �ؼ��� �ſ� ����
#   - �����ǻ�����̹Ƿ� �������� �Ҿ��ϰų� overfit�ɼ� ����

# [ iris data set - DT ]
install.packages('rpart')
library(rpart)

# 1. sampling
v_rn <- sample(1:nrow(iris), size = nrow(iris) * 0.7)
iris_train <- iris[v_rn, ]
iris_test  <- iris[-v_rn, ]

# 2. �� ����(training, pattern ����)
m_dt <- rpart(data=iris_train, formula = Species ~ .)

# �� Ȯ��(�� ����� �з��� �� ���������� ���ð���)
m_dt

# 3. �� �ð�ȭ
dev.new()
plot(m_dt, compress = T)
text(m_dt, cex=1.5)

install.packages('rpart.plot')
library(rpart.plot)

prp(m_dt, type=4, extra=2, digits = 3)

# 4. �� ��
v_y <- iris_test$Species
v_predict <- predict(m_dt, newdata=iris_test, type = 'class')

sum(v_y == v_predict) / nrow(iris_test) * 100

# ���� ���� ���ο� �����Ϳ� ���� Y�� ���� 
iris_new = data.frame(Sepal.Length=4.7, 
                      Sepal.Width=3.8, 
                      Petal.Length=1.9, 
                      Petal.Width=0.3)
# 5. �� Ʃ��
# 1) �Ű����� �� Ȯ��
m_dt$control

# 2) �Ű����� ����
# 1-1) minbucket
# - �߰� ���� ��带 �����ϱ� ���� ���з� �������� �ּ� ����
# - ���� 7�� ��� ���з� �����Ͱ� 7�̸��� ���� ���̻� �з����� X
# - ���� �������� ���� ���⵵�� Ŀ��

# 1-2) minsplit
# - minbucket = round(minsplit/3)

# 1-3) cp
# 1-4) maxdepth

# 3) �Ű����� ���� 
m_dt2 <- rpart(data=iris_train, 
               formula = Species ~ . ,
               control = rpart.control(minbucket=2))

m_dt2

# 3) ������ �Ű����� �� Ȯ��
# [ ���� ���� ]
# minbucket�� 1~10 ��ȭ�ɶ��� ������ ��ȭ Ȯ��
# overfit ����  
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
  
  # ����ȭ
  v_tr_score <- c(v_tr_score, tr_score)
  v_te_score <- c(v_te_score, te_score)
}

# �ð�ȭ
dev.new()
plot(1:10, v_tr_score, type = 'o', col='blue', ylim = c(90,100))
lines(1:10, v_te_score, type = 'o', col='red')


# ** overfit(��������) : train data set�� ���� ������ �����ϰ� 
# ���ϰ� ������ ���� ���� �������� ���ο� ������(test data)�� ����
# �����°��� ���̰� �� �߻��ϴ� ��츦 �ǹ�


# 6. ���ο� �����Ϳ� ���� ����
predict(m_dt, newdata=iris_new, type = 'class')

# [ ���� ���� ]
# cancer.csv ������ ȯ�ڵ��� �� ������ �����͸� ����,
# �̸� ���� �缺/�Ǽ��� �з��� �� �ִ� ���� ����(decision tree)
cancer <- read.csv('cancer.csv')
str(cancer)

# ���� �������� ���� �� ���������� �ǹ� �ľ�
dev.new()
plot(cancer[, 3:11], col = cancer$diagnosis)

# 2) Random Forest   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

     