# [ Ʈ�� ��� ���� ���� ���� ��� ]
Ʈ����� ���� �� �� ������ �߿䵵�� �ľ�,
�ش� ���� �� �߿��� �������� �����Ͽ� ��ġ�ϴ� ����� ����
=> �ٸ� �𵨿� ���� ���� ���ÿ� ���� �δ��� ����
=> final model�� ������� �ʴ��� ���� ���� �� ���
(�� ��� ���� ���� ���)

# �Ҽ���
���� ��忡 ���� �ٸ� class�� �����Ǿ� �ִ� ������ ��Ÿ�� ��ġ
��ǥ���� �Ҽ��� ���� ���� => "���� �Ҽ���"

# ���� �Ҽ��� ���� ���(Y�� 2���� class�� �����Ǿ� �ִ� ���)
f(p) = p(1-p), p�� Ư�� class�� ���� Ȯ���ǹ�

ex)  (OOXXX)�� ������ ����� ���ϺҼ��� ���
p=O�� ���õ� Ȯ���̶� ����, p=2/5, 1-p=3/5
f(p) = 2/5 * 3/5 = 0.24


# Ʈ����� �� ������ ���� �߿䵵 Ȯ��
m_dt$variable.importance

Petal.Width Petal.Length Sepal.Length  Sepal.Width 
61.20231     58.32399     35.90656     22.37384

30/5 => 97%
30   => 98%

# ���Ǻ� �߷� ����(ctree)
# ���� �ǻ���������� �� ����� ���ÿ� ����� ���Ǽ� ������
# �� �� ���ٴ� ������ ����

install.packages('party')
library(party)

m1 <- ctree(formula = Species ~ . , 
            data=iris)
m1          # �� Ȯ��

dev.new()
plot(m1)

# �м� �� ��Ÿ ��������
# 1. Sampling �Ҷ����� �ٸ� ���� ����, ������ �޶����� ����
# => cross validation(CV, ��������)

# 2. ���� ����, ���� ����
# X1X2, (X1 + X2)/2, log(x1)


# 2) Random Forest   
# - decision tree�� �����ǻ���� ����� ����, �ټ��� tree ����
# - �������� tree�� �����Ͽ� ���� ����� ���� ���
# - randomForest-regressor, randomForest-classfier ����
# - ��� �Ǵ� ��ǥ�� ���� ���� ���
# - ���� �ٸ� Ʈ���� ����

install.packages('randomForest')
library(randomForest)

# 1) sampling
v_rn <- sample(1:nrow(iris), size = nrow(iris) * 0.7)
tr_iris <- iris[v_rn, ]
te_iris <- iris[-v_rn, ]

# 2) training
m_rf <- randomForest(Species ~ ., data=tr_iris)

# Ư�� �߿䵵 Ȯ��
m_rf$importance

#                MeanDecreaseGini
# Sepal.Length         6.637587
# Sepal.Width          1.122958
# Petal.Length        35.381573
# Petal.Width         26.070911

# 3) score
v_predict <- predict(m_rf, newdata = te_iris, type = 'class')
sum(te_iris$Species == v_predict) / nrow(te_iris) * 100

# 4) predict(Ȯ���� �������� �Ϻ� ���� �� ���� ����)
v_rn2 <- sample(1:nrow(te_iris), size = 1)
v_newdata <- te_iris[v_rn2, ]   

predict(m_rf, newdata = v_newdata)   # ������
v_newdata$Species                    # ������

# 5) �Ű����� Ʃ��
# 5-1) ntree : Ʈ���� ����
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
     main='RF������ ntree��ȭ�� ���� ������ ��ȭ')


# 5-2) mtry 
# - �� ��忡 �������� ���ý� �����Ǵ� �ĺ��� ����
# - ���� �ٸ� Ʈ���� �����ϱ����� ���� �Ű�����
# - Ŭ���� ���� ���������� ����, ���� ����� Ʈ�� ����,
# - �������� ��ü �� �����ϰ� ���õ� �Ϻ� ���������� ����,
#   ���� �ٸ� Ʈ�� ����, �� Ʈ���� �ſ� �������� ���ɼ� 

# [ ���� ���� ]
# mtry ���� ��ȭ�� ���� ������ �� ���������� ��ȭ Ȯ��
v_score_tr <- c() ; v_score_te <- c()

for (i in 1:4) {
  # �� ����
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
     main='RF������ mtry��ȭ�� ���� ������ ��ȭ')
lines(1:4, v_score_tr, type='o', col='blue')


# [ ���� - random forest�� ȸ�� ]
data("airquality")
airquality

str(airquality)  # Ozone : ������(Y), ������ �������� ����
m_rf_reg <- randomForest(Ozone ~ . , 
                         data = airquality,
                         mtry = 3,
                         na.action = na.omit)
m_rf_reg



mean((12000 - 15000)^2 , (13000 - 13100)^2)
mean((12 - 15)^2 , (13 - 13)^2)


# Tree ��� ��
Decision Tree(DT) -> Random Forest(RF)
-> Gradiant Boosting Tree(GB)
-> eXtreme Gradiant Boosting Tree(XGB)

rpart(Y ~ X, data=)        # Y�� non-factor���� ����
randomForest(Y ~ X, data=) # Y�� factor���� ����

# �Ÿ� ��� ��
# - �� ����ġ�� �Ÿ��� ���, ���� ����� �������� �ൿ�� �״�� ��õ
# - ��õ�ý���, �̹��� �ν� �ý����� �⺻�� �Ǵ� �˰�����
# - �Ÿ��� ������ ������ �ݵ�� �ʿ�
# - �̻�ġ�� �ſ� �ΰ��� ��
 





