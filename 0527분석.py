# =============================================================================
# # 교호작용 기법(interaction)
# =============================================================================
 - 변수간의 결합*
 - 설명변수끼리의 상호곱 변수 생성, 그 중 의미있는  변수를 추출하기 위함
 - 적은 설명변수로 많은 설명변수로의 확장 가능 (설명변수 추가 없이)
 - 초기 변수 연구 단계에서 고려되는 기법
 => 수업 목표 : "의미있는 교호작용(변수간 결합)을 찾는 것이 목적"
 
아파트 가격 <= 지하주차장 면적 * 평균강수량
아파트 가격 <= 지하주차장 면적 + ... +

run profile1
from sklearn.preprocessing import PolynomialFeatures

# interaction - iris data
# 1.데이터 로딩
df_iris = iris()

# 2.데이터 분리
train_x,test_x,train_y,test_y = train_test_split(df_iris['data'],
                                                 df_iris['target'],
                                                 random_state=0)

# 3. interaction 모델 생성
m_poly = PolynomialFeatures(degree=2)  # degree=2, 2차원

# 4. 모델에 데이터 학습 -> 설명변수 변환기준 찾기
 #fitting : (확장된 모델알려준다)변수명 추출
m_poly.fit(train_x)

# 5. 실 데이터 변환 (교호작용 기준에 맞게 설명변수 변환)
# transform : 확장된 형식에 맞춰 리폼(실데이터에 대한 변환)
train_x_poly = m_poly.transform(train_x)

# 6. 변환된 변수 형태 확인
m_poly.get_feature_names()                           # 실제 변수명X
m_poly.get_feature_names(설명변수)                   # 설명변수명 출력  
m_poly.get_feature_names(df_iris['feature_names'])  # 실제 변수명으로 추출

['sepal length', 'sepal width', 'peral length', 'petal width']

            fitting
y  x1  x2   =>   y x1  x2  x1^2  x2^2  x1x2
    1   2           1   2    1     4     2 
    2   3           2   3    4     9     6 
   
transform : 확장된 형식에 맟줘 리폼(실데이터에 대한 변환)
fitting : (확장된 모델알려준다)변수명 추출


# 7. 확장된 데이터셋으로 knn모델 적용
m_knn=knn(n_neighbors=3)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)  # 4개 설명 변수 예측값 - 97.37%

# 다시보기
m_knn2=knn(n_neighbors=3)
m_knn2.fit(train_x_poly, train_y)
test_x_poly = m_poly.transform(test_x)
m_knn2.score(test_x_poly, test_y)  # 확장된 설명변수 예측값 - 97.37%

# 8. 확장된 설명변수 중 의미있는 교호작용 추출

m_rf = rf()
m_rf.fit(train_x_poly, train_y)

df_iris_poly_col = m_poly.get_feature_names(df_iris['feature_names'])
s1 = Series(m_rf.feature_importances_, index = df_iris_poly_col)
s1.sort_values(ascending=False)













