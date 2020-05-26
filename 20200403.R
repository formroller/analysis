# 특성공학
# - 데이터 분석시 수행하는 변수의 다양한 시각적 연구
# - 변수의 선택 및 제거
# - 변수 결합
# - 변수 변형


# 분석에서의 변수의 선택(feature selection)
# 1. 통계적 접근
#  1) 전진선택법(Forward Selection) : 변수를 계속 추가하는 과정
#    Y ~ .
#    Y ~ X1
#    Y ~ X1 + X2
   
#  2) 후진선택법(Backward Selection) : 전체 변수를 학습 후 하나씩 제거
#    Y ~ X1 + X2 + ... + Xn
#    Y ~      X2 + ... + Xn
#    Y ~           X3 ... + Xn
    
#  3) stepwise기법
#    Y ~ X1 + X2 + ... + Xn
#    Y ~      X2 + ... + Xn        # X1제거
#    Y ~           X3  + ... + Xn  # X2제거
#    Y ~ X1 +    + X3  + ... + Xn  # X2제거 시 X1의 추가 고려

# 예제    
install.packages('mlbench')
library(mlbench)
data(BostonHousing)   

str(BostonHousing)   
   
# 2. 모델적 접근 : 머신러닝/딥러닝 모델 내부에 변수선택에 대한 내용 포함
#    ex) DT, RF, NN
# 
# Y ~ X1 + X2 + X3 + .... + Xn
# 
