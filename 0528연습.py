# =============================================================================
# (SVM) cancer data
# =============================================================================
run profile1
from sklearn.svm import SVM

#-- 스케일링 / SVM
cancer = cancer()

cancer_tr_x, cancer_te_x, cancer_tr_y, cancer_te_y = train_test_split(cancer.data,
                                                                      cancer.target,
                                                                      random_state=0)
m_mms = MinMaxScaler() # MinMaxScaler, 스케일링
m_mms.fit(cancer_tr_x)

# 그 이유는 각 독립 변수들이 0.1 단위부터 수백 단위까지 제각각의 크기를 가지고 있기 때문이다.
# 조건수가 크면 약간의 오차만 있어도 해가 전혀 다른 값을 가진다. 따라서 조건수가 크면 회귀분석을 사용한 예측값도 오차가 커지게 된다.
# 따라서 스케일링 작업 실시

cancer_tr_x_sc = m_mms.transform(cancer_tr_x)
cancer_te_x_sc = m_mms.transform(cancer_te_x)

m_svm = SVC()
m_svm.fit(cancer_tr_x_sc, cancer_tr_y)
m_svm.score(cancer_te_x_sc, cancer_te_y)  # 95.1%

# -- 매개변수 튜닝
v_c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
v_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

col_c =[]; col_gamma =[]; col_train_score =[]; col_test_score=[];
for i in v_c:
    for j in v_gamma:
        m_svm = SVC(C = i, gamma = j)
        m_svm.fit(cancer_tr_x_sc, cancer_tr_y)
        col_c.append(i)
        col_gamma.append(j)
        col_train_score.append(m_svm.score(cancer_tr_x_sc, cancer_tr_y))
        col_test_score.append(m_svm.score(cancer_te_x_sc, cancer_te_y))


df_svm_score = DataFrame({'C':col_c,
                          'gamma':col_gamma,
                          'train_score':col_train_score,
                          'test_score':col_test_score})

df_svm_score.sort_values(by=['train_score','test_score'], ascending = False)
df_score = df_svm_score.sort_values(by=['test_score','train_score'], ascending = False)

# -- 시각화
df_score.iloc[:,2:4]

df_score.index = np.arange(df_score.shape[0])  # index 조정
df_score.iloc[:,2:4].plot()

f2 = lambda x : 'C : ' + str(x[0]) + 'gamma : ' + str(x[0])
v_text = df_score.apply(f2,axis=1)

plt.xticks(df_score.index,
           v_text,
           rotation = 90,
           fontsize=8)

#-- heatmap
import mglearn
a1 = np.array(df_svm_score.test_score).reshape(7,7)
mglearn.tools.heatmap(a1, 'gamma', 'C', v_gamma, v_c, cmap='viridis')

# =============================================================================
# (PCA + knn) iris data
# =============================================================================
from sklearn.decomposition import PCA

iris = iris()
iris_tr_x, iris_te_x, iris_tr_y, iris_te_y = train_test_split(iris.data,
                                                              iris.target,
                                                              random_state=0)
#-- StanddardScaler
m_sc = StandardScaler()
m_sc.fit(iris_tr_x)
train_x_sc = m_sc.transform(iris_tr_x)
test_x_sc = m_sc.transform(iris_te_x)

m_pca = PCA(n_components = 2)
m_pca.fit(train_x_sc)

train_x_pca = m_pca.transform(train_x_sc)
test_x_pca = m_pca.transform(test_x_sc)

#-- 시각화
mglearn.discrete_scatter(train_x_pca[:,0],
                         train_x_pca[:,1],
                         iris_tr_y)

# a)knn 적용
m_knn = knn(n_neighbors = 3)
m_knn.fit(train_x_pca, iris_tr_y)
m_knn.score(test_x_pca, iris_te_y)  # 89.47%

# b) RF 적용
m_rf = rf()
m_rf.fit(train_x_pca, iris_tr_y)
m_rf.score(test_x_pca, iris_te_y)  # 85.81%

# c)DT 적용
m_dt = dt()
m_dt.fit(train_x_pca, iris_tr_y)
m_dt.score(test_x_pca, iris_te_y)  # 89.47%

# d) SVM 적용
m_svm = SVC()
m_svm.fit(train_x_pca, iris_tr_y)
m_svm.score(test_x_pca, iris_te_y)  # 89.47%

# scaling + interacion + pca   * interaciton 과정 추가

m_poly = PolynomialFeatures(degree=2)
m_poly.fit(train_x_sc)

train_x_poly = m_poly.transform(train_x_sc)
test_x_poly = m_poly.transform(test_x_sc)

m_pca = PCA(n_components = 2)
m_pca.fit(train_x_poly, iris_tr_y)

iris_tr_x_spp = m_pca.transform(train_x_poly)
iris_te_x_spp = m_pca.transform(test_x_poly)

# -> knn
m_knn = knn(n_neighbors=3)
m_knn.fit(iris_tr_x_spp, iris_tr_y)
m_knn.score(iris_te_x_spp, iris_te_y)  # 89.47 -> 86.84
# -> RF
m_rf = rf()
m_rf.fit(iris_tr_x_spp, iris_tr_y)
m_rf.score(iris_te_x_spp, iris_te_y)   # 85.81 -> 94.8%

# -> DT
m_dt = dt()
m_dt.fit(iris_tr_x_spp, iris_tr_y)
m_dt.score(iris_te_x_spp, iris_te_y)   # 89.47 -> 86.84%

# -> SVM
m_svm = SVC()
m_svm.fit(iris_tr_x_spp, iris_tr_y)
m_svm.score(iris_te_x_spp, iris_te_y)  # 89.47 -> 89.47%

# =============================================================================
# (PCA + SVM) cancer data
# =============================================================================
#1) SVM
df_cancer = cancer()
cancer_tr_x, cancer_te_x, cancer_tr_y, cancer_te_y = train_test_split(cancer.data.
                                                                      cancer.target,
                                                                      random_state=0)
m_mms = MinMaxScaler()
m_mms.fit(cancer_tr_x)

cancer_tr_x_sc = m_mms.transform(cancer_tr_x)
cnacer_te_x_sc = m_mms.transform(cancer_te_x)

m_svm = SVC()
m_svm.fit(cancer_tr_x_sc, cancer_tr_y)
m_svm.score(cancer_te_x_sc, cancer_te_y)  # 95.1%

#-- 매개변수 튜닝
m_svm
v_c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
v_gamma = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

col_c=[]; col_gamma=[]; col_train_score=[]; col_test_score=[]

for i in v_c :
    for j in v_gamma :
        m_svm = SVC(C = i, gamma = j)
        m_svm.fit(cancer_tr_x_sc, cancer_tr_y)
        col_c.append(i)
        col_gamma.append(j)
        col_train_score.append(m_svm.score(cancer_tr_x_sc, cancer_tr_y))
        col_test_score.append(m_svm.score(cancer_te_x_sc, cancer_te_y))
        
df_svm_score = DataFrame({'C' : col_c,
                          'gamma': col_gamma,
                          'train_score':col_train_score,
                          'test_score':col_test_score})
df_svm_score.sort_values(by = ['train_score','test_score'], ascending = False)
df_score = df_svm_score.sort_values(by = ['test_score', 'train_score'], ascending = False)

df_score.index = np.arange(df_score.test_score.shape[0])
df_score

# 2) PCA + SVM
m_pca = PCA(n_components = 2)
m_pca.fit(cancer_tr_x_sc)

cancer_tr_x_scP = m_pca.transform(cancer_tr_x_sc)
cancer_te_x_scP = m_pca.transform(cancer_te_x_sc)

m_svm_pca = SVC()
m_svm_pca.fit(cancer_tr_x_scP, cancer_tr_y)
m_svm_pca.score(cancer_te_x_scP, cancer_te_y)  # 92.3%

v_c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
v_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

cancer_c=[]; cancer_gamma=[]; cancer_train=[]; cancer_test=[]
for i in v_c:
    for j in v_gamma:
        m_svm = SVC(C = i, gamma = j)
        m_svm.fit(cancer_tr_x_scP, cancer_tr_y)
        cancer_c.append(i)
        cancer_gamma.append(j)
        cancer_train.append(m_svm.score(cancer_tr_x_scP, cancer_tr_y))
        cancer_test.append(m_svm.score(cancer_te_x_scP, cancer_te_y))

cacner_svm_score = DataFrame({'C' : cancer_c,
                              'gamma':cancer_gamma,
                              'train_scroe':cancer_train,
                              'test_score':cancer_test})
    
mglearn.discrete_scatter(cancer_tr_x_scP[:,0],
                         cancer_tr_x_scP[:,1],
                         cancer_tr_y)
