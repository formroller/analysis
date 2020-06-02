run profile1

# =============================================================================
# # cancer data - 그리드 서치 + CV
# =============================================================================

# 1) data loading
df_cancer = cancer()
df_cancer_x = df_cancer.data
df_cancer_y = df_cancer.target

# 2) data split
x_trainval,x_test,y_trainval,y_test = train_test_split(df_cancer_x,
                                                        df_cancer_y,
                                                        random_state=0)

x_train, x_val, y_train, y_val = train_test_split(x_trainval,
                                                  y_trainval,
                                                  random_state=0)

# 3) SVM에서의 매개변수 튜닝
#3-1) cv 수행 없이 매개변수 선택에 대한 모델 평가
# 매개변수 선택 과정
best_score=0
for v_c in [0.001, 0.01, 0.1, 1, 10, 100] :
    for v_gamma in [0.001, 0.01, 0.1, 1, 10, 100] :
        m_svm = SVC(C = v_c, gamma = v_gamma)
        m_svm.fit(x_train,y_train)
        vscore = m_svm.score(x_val,y_val)
        
        if vscore > best_score :
            best_score = vscore
            bset_params = {'C':v_c, 'gamma':v_gamma}
            
m_svm = SVC(**bset_params)  # 최종 변수 전달
m_svm.fit(x_trainval,y_trainval) # 매개변수 재학습

m_svm.score(x_test,y_test) # 재학습 후 적용

#3-2) 매개변수 선택에 대한 모델 평가시 cv고려
for v_c in [0.001, 0.01, 0.1, 1, 10, 100] :
    for v_gamma in [0.001, 0.01, 0.1, 1, 10, 100] :
        m_svm = SVC(C = v_c, gamma = v_gamma)
        cv_score = cross_val_score(m_svm, x_trainval,y_trainval,cv= 5)  ### 자체가 내부에서 train/test 나눈다
        
        v_score = np.mean(cv_score)
        
        if vscore > best_score :
            best_score = vscore   # 다시보기!!
            best_params = {'C':v_c, 'gamma':v_gamma} 

m_svm = SVC(**best_params)
m_svm.fit(x_trainval, y_trainval)

m_svm.score(x_test, y_test)

# =============================================================================
# # GridSearchCV
# =============================================================================
 - 위에서 직접 작성한 그리드서치시 CV를 고려하는 모델 (함수)
 
from sklearn.model_selection import GridSearchCV

# 1. 학습시킬 모델 생성
m_svm = SVC()

# 2. 그리드 서치로 찾을 매개변수 조합 생성
v_parameter = [{'C':[0.001, 0.01, 0.1, 1, 10, 100],
                'gamma':[0.001, 0.01, 0.1, 1, 10,100]}]

# 3. GridSearchCV모델 생성
m_gridcv = GridSearchCV(m_svm,        # 학습 모델
                        v_parameter,  # 학습 모델의 배개변수(딕셔너리 형태)
                        cv = 5)       # 교차검증 횟수

# 4. GridSearchCV모델 학습
m_gridcv.fit(x_trainval, y_trainval)

# 5. 매개변수 확인
m_gridcv.best_params_   # 최적의 매개변수 조합 확인
m_gridcv.best_score_    # 매개변수 선택 시 best_score(CV포함)

df_result = DataFrame(m_gridcv.cv_results_)   # 교차검증 과정 확인
df_result.dtypes  # split0_test_score    float64 매개변수 테스트(validation score/교차검증)
                  # mean_test_score      float64 5개의 평균  


df_result2 = df_result.iloc[:,[4,5,7,8,9,10,11,12]]
df_result2.sort_values(by='mean_test_score', ascending = False)
# 6. 최종모델 평가
m_gridcv.score(x_test, y_test)  # m_gridcv 모델에 최적의 매개변수 저장
                                # 선택된 매개변수로 재학습 포함    
                                
# 7. 그리드서치 결과 시각화
arr_score = np.array(df_result.mean_test_score).reshape(6,6)

mglearn.tools.heatmap(arr_score,
                      xlabel = 'gamma',
                      xtickslabels = v_parameter['gamma'],
                      ylabel = 'C',
                      ytickslabels = v_parameter['C'],
                      cmap='viridis')

# =============================================================================
# 보스턴 주택 가격 데이터셋(회귀)
# =============================================================================
# 회귀분석
 - 종속변수와 설명변수의 인과관계 분석
 - 각 설명변수마다 회귀계수를 특정, 회귀계수의 방향 및 크기로 설명변수의 종속변수에의 영향력 판단
 - 여러가지 통계적 가설에 의해 생성된 통계적 모델이므로 모델의 예측력을 높이기 위해 여러가지 가설 검증이 필요
 - 최근에는 전통회귀보다 분류모델의 회귀나 KNN 사용하는 추세
 
 Ridge / Lasoo = 전통회귀의 다중공성선 해결하기 위해 고안된 방법


from sklearn.datasets import load_boston as boston
df_boston = boston()
df_boston.feature_names
df_boston.data.shape     # (506,13)

# 상호작용 고려된 형태의 보스턴 주택 가격 데이터 셋(회귀)
X,y = mglearn.datasets.load_extended_boston??
X,y = mglearn.datasets.load_extended_boston()

X.shape                 # (506,104) -> 13개의 데이터셋에서 104개의 설명변수로 확장

# 1. 선형 / 비선형 회귀
from sklearn.linear_model import LinearRegression

#1. 데이터 로딩
#2. 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(df_boston.data,
                                                    df_boston.target,
                                                    random_state=0)

#3. 데이터 학습
m_reg = LinearRegression()
m_reg.fit(train_x, train_y)
m_reg.score(train_x, train_y)  # R^2 = 76.97%
m_reg.score(test_x, test_y)    # R^2 = 63.5%

#4. 회귀계수 확인
m_reg.coef_         # 각 설명변수 계수
m_reg.intercept_    # 절편


#[extended boston data set으로 회귀 수행]
X,y = mglearn.datasets.load_extended_boston()

# data split()
train_X,test_X, train_Y, test_Y = train_test_split(X,
                                                   y,
                                                   random_state=0)
m_reg2 = LinearRegression()
m_reg2.fit(train_X, train_Y)
m_reg2.score(train_X,train_Y)  # R^2 = 95.2%
m_reg2.score(test_X,test_Y)    # R^2 = 60.74%
#=> 회귀는 설명변수 많아질 수록 설명력이 높아진다(interacton시 분류보다 좋은 결과 도출)

m_reg2.coef_
(abs(m_reg2.coef_) <0.1).sum()  # 0개
m_reg2.intercept_               # 30.93

# =============================================================================
# # 릿지(Ridge)
# =============================================================================
 - 설명변수중 중요도가 떨어지는 설명변수의 회귀계수를 0에 가깝게 만드는 모델
 - 각 회귀계수의 가중치를 다르게 부여, 변수의 중요도를 떨어트리면서 모델의 복잡도 제어
 - alpha라는 매개변수의 크기에 따라 모델 복잡도를 선택할 수 있음.
 
# [새로운 데이터의 예측이 위보다 높아짐. 더 좋은 모델]
m_ridge = Ridge()
m_ridge.fit(train_X, train_Y)
m_ridge.score(train_X, train_Y)   # R^2 = 88.58%
m_ridge.score(test_X, test_Y)     # R^2 = 75.28%

(abs(m_ridge.coef_) < 0.1).sum()       # 6개

# alpha값의 변화에 따른 모델 변화 확인
m_ridge = Ridge(alpha = 0.01)
m_ridge.fit(train_X, train_Y)
m_ridge.score(train_X, train_Y)   # R^2 = 94.45%
m_ridge.score(test_X, test_Y)     # R^2 = 70.22%

(abs(m_ridge.coef_) < 0.1).sum()       # 1개
=> 모델이 더 복잡해졌다.

# [alpha = 10]
m_ridge = Ridge(alpha = 10)
m_ridge.fit(train_X, train_Y)
m_ridge.score(train_X, train_Y)   # R^2 = 78.82%
m_ridge.score(test_X, test_Y)     # R^2 = 63.59%

(abs(m_ridge.coef_) < 0.1).sum()       # 6개
==> alpha 값에 따라 모델 달라진다.

# 최적의 alpha값 확인 

alpha1 = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_score=[]; test_score=[]  # 다시보기
for i in[0.001, 0.01, 0.1, 1, 10, 100, 1000]: 
    m_rideg = Ridge(alpha = i)
    m_ridge.fit(train_X, train_Y)
    train_score.append(m_ridge.score(train_X,train_Y))
    test_score.append(m_ridge.score(test_X, test_Y))

plt.plot(train_score, label='train_score')
plt.plot(test_score, label='test_score')
plt.xticks(np.arange(6), [0.001, 0.01, 0.1, 1, 10, 100, 1000])
plt.legend()
# =============================================================================
# 라쏘(Lasso)
# =============================================================================
X,y = mglearn.datasets.load_extended_boston


m_lasso= Lasso()
m_lasso.fit(train_X, train_Y)     
m_lasso.score(train_X, train_Y)   # R^2 = 29.32%
m_lasso.score(test_X, test_Y)     # R^2 = 20.93%

(m_lasso.coef_ == 0).sum()       # 100개 탈락시킨다.

# alpha 변화에 따른 모델 변화
m_lasso= Lasso(alpha=0.1)
m_lasso.fit(train_X, train_Y)     
m_lasso.score(train_X, train_Y)   # R^2 = 77.1%
m_lasso.score(test_X, test_Y)     # R^2 = 63.02%

(m_lasso.coef_ == 0).sum()       # 96개


# StatsModels 패키지을 사용한 회귀분석
import statsmodels.api as sm

X,y = mglearn.datasets.load_extended_boston()
m_ols = sm.OLS(y,X).fit()

print(m_ols.summary())  # 회귀계수, 회귀오차, t통계량, 유의확률, 신뢰구간


H0 : a1  = 0
H1 : a1 != 0

a1 = H0에 대한 회귀계수

#유의수준이 0.05일때 
P-value < 0.05  *H0 기각 => a1 계수의 의미가 있다.

#[당뇨병 데이터셋 - 수치 예측 (회귀)]
from sklearn.datasets import load_diabetes
diabetes = diabetes()

diabetes.data.shape   # (442,10)
diabetes.target

diabetes.feature_names  # 설명변수 명

print(diabetes.DESCR)
