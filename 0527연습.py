교호작용 : 변수간의 결합
 - 초기 변수 연구 단계에서 고려되는 기법
 
 * 목적 : 의미있는 교호작용(변수간 결합)을 찾는 것.
 
run profile1
from sklearn.preprocessing import PolynomialFeatures as pf

# =============================================================================
#  interaction - iris
# =============================================================================
df_iris = iris()
train_x, test_x, train_y, tset_y = train_test_split(df_iris.data,
                                                    df_iris.target,
                                                    random_state=0)

m_ploy = PolynomialFeatures(degree=2)
m_poly.fit(train_x)

train_x_poly = m_poly.transform(train_x)
test_x_poly = m_poly.transform(test_x)

m_poly.get_feature_names(df_iris.feature_names)

m_knn = knn(n_neighbors = 3)
m_knn.fit(train_x, train_y)
m_knn.score(test_x, test_y)  # 97.3%

m_knn2=knn(n_neighbors = 3)
m_knn2.fit(train_x_poly, train_y)
m_knn2.score(test_x_poly, test_y) # 97.3%

m_rf = rf()
m_rf.fit(train_x_poly, train_y)
m_rf.importances_

iris_poly_col = m_poly.get_feature_names(df_iris.feature_names)
s1 = Series(m_rf.feature_importances_, index = iris_poly_col)
s1.sort_values(ascending = False)[:10]

# =============================================================================
# interaction - cancer
# =============================================================================
cancer = cancer()

train_x, test_x, train_y, test_y = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)

poly_m = PolynomialFeatures(degree=2)

poly_m.fit(train_x)

tr_x_poly = poly_m.transform(train_x)
te_x_poly = poly_m.transform(test_x)

poly_m.get_feature_names(cancer.feature_names)

rf_m = rf()
rf_m.fit(tr_x_poly, train_y)

cancer_poly_col = poly_m.get_feature_names(cancer.feature_names)
s1 = Series(rf_m.feature_importances_, index = cancer_poly_col)
v1_importance = s1.sort_values(ascending = False)[:30]

plt.barh(range(30), v1_importance, align='center')
plt.yticks(np.arange(30), v_importance.index)

# =============================================================================
# scaling - iris
# =============================================================================
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

iris = iris()
train_x, test_x, train_y, test_y = train_test_split(iris.data,
                                                    iris.target,
                                                    random_state=0)
m_ms = MinMaxScaler()
m_ms.fit(train_x)

tr_x_sc = m_ms.transform(train_x)
te_x_sc = m_ms.transform(test_x)

train_x_sc.min(axis=0)
test_x_sc.min(axis = 0)

train_x_sc.max(axis = 0)
test_x_sc.max(axis = 0)

fig,axes = plt.subplots(1,2)

axes[0].scatter(train_x[:,0], train_x[:,1],
                c = mglearn.cm2(0), label = 'train', s=60)
axes[0].scatter(test_x[:,0], test_x[:,1],
                c = mglearn.cm2(1), label = 'test', s=60)
axes[0].legend()

axes[1].scatter(tr_x_sc[:,0], tr_x_sc[:,1],
                c = mglearn.cm2(0), label = 'train_scale', s=60)
axes[1].scatter(te_x_sc[:,0], te_x_sc[:,1],
                c = mglearn.cm2(3), label = 'test_scale', s=60)
axes[1].legend()

# scaling - cancer 
cancer = cancer()

train_x, test_x, train_y, test_y = train_test_split(cancer.data
                                                    cancer.target,
                                                    random_state=0)

m_ms = MinMaxScaler()
m_ms.fit(train_x)

tr_x_sc = m_ms.transform(train_x)
te_x_sc = m_ms.transform(test_x)

tr_x_sc.min(axis=0)
tr_x_sc.max(axis=0)

te_x_sc.min(axis=0)
te_x_sc.max(axis=0)

m_poly = PolynomialFeatures(degree=2)
m_poly.fit(tr_x_sc)
train_x_poly = m_poly.transform(tr_x_sc)
test_x_poly= m_poly.transform(te_x_sc)

cancer_po_col = m_poly.get_feature_names(cancer.feature_names)

# Random Forest
m_rf = rf()
m_rf.fit(train_x_poly, train_y)
m_rf.score(test_x_poly, test_y)  # 95.8%
