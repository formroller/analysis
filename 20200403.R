# Ư������
# - ������ �м��� �����ϴ� ������ �پ��� �ð��� ����
# - ������ ���� �� ����
# - ���� ����
# - ���� ����


# �м������� ������ ����(feature selection)
# 1. ����� ����
#  1) �������ù�(Forward Selection) : ������ ��� �߰��ϴ� ����
#    Y ~ .
#    Y ~ X1
#    Y ~ X1 + X2
   
#  2) �������ù�(Backward Selection) : ��ü ������ �н� �� �ϳ��� ����
#    Y ~ X1 + X2 + ... + Xn
#    Y ~      X2 + ... + Xn
#    Y ~           X3 ... + Xn
    
#  3) stepwise���
#    Y ~ X1 + X2 + ... + Xn
#    Y ~      X2 + ... + Xn        # X1����
#    Y ~           X3  + ... + Xn  # X2����
#    Y ~ X1 +    + X3  + ... + Xn  # X2���� �� X1�� �߰� ����

# ����    
install.packages('mlbench')
library(mlbench)
data(BostonHousing)   

str(BostonHousing)   
   
# 2. ���� ���� : �ӽŷ���/������ �� ���ο� �������ÿ� ���� ���� ����
#    ex) DT, RF, NN
# 
# Y ~ X1 + X2 + X3 + .... + Xn
# 