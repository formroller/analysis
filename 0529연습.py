run profile1
df1 = pd.read_csv('ThoraricSurgery.csv', header = None)

df1_x = df1.iloc[:,1:17]
df1_y = df1.iloc[:,17]

train_x, test_x, train_y, test_y = train_test_split(df1_x,
                                                    df1_y,
                                                    random_state = 0)

m_svc = SVC()
m_svc.fit(train_x.values, train_y.values)
m_svc.score(test_x.values, test_y.values)  # 90.68%

# =============================================================================
# (PCA + KNN) 이미지 분석
# =============================================================================
#1) data set 추출
from sklearn.datasets from fetch_lfw_people

# 다운로드
human = fetch_lfw_people(min_faces_per_person=30, resize=0.7)
# data set 구성
human.keys()
human.images.shape #(2208, 87, 65)
human.data.shape   #(2208, 5655)
#data set 시각화
fig,ax = plt.subplots(1,3, figsize = (15,8))

ax[0].imshow(human.images[11])
ax[1].imshow(human.images[110])
ax[2].imshow(human.images[1100])

name1 = human.target_names[people.target[11]]
name2 = human.target_names[people.target[110]]
name3 = human.target_names[people.target[1100]]

ax[0].set_title(name1)
ax[1].set_title(name2)
ax[2].set_title(name3)

# 2. Data Sampling(사람별 최대 50개 이미지 선택)
np.where(human.target == 61)

v_row=[]
for i in np.unique(human.target) : 
    v_row = v_row + list(np.where(human.target == i)[0][:50])

x_human = human.data[v_row]
y_human = human.target[v_row]

# 3. scaling
x_human_sc = x_human/255

# 4. train, test set 분리
train_x, test_x, train_y, test_y = train_test_split(x_human_sc,
                                                    y_human,
                                                    random_state=0)
# 5. knn 학습
m_knn = knn(n_neighbors = 3)
m_knn.fit(train_x, train_y)

# 6. knn 모델 평가
m_knn.score(test_x, test_y)  # 26.92%

# 7. PCA기법으로 인공변수 유도 (n_components = 100)
m_pca = PCA(n_components = 100, whiten = True)
m_pca.fit(train_x)

human_tr_x_pca = m_pca.transform(train_x)
human_te_x_pca = m_pca.transform(test_x)

# 8. 인공변수(100개)로 knn 모델 학습 및 평가
m_knn = knn(n_neighbors = 3)
m_knn.fit(human_tr_x_pca, train_y)
m_knn.score(human_te_x_pca, test_y)  # 35.9%

# =============================================================================
#  CV : Cross Validation (교차검증)
# =============================================================================
 - 평가점수 일반화
#1) 데이터 로딩
iris = iris()
#2) k-fold 수행
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

m_kfold = KFold(n_splits=5, shuffle = True, random_state = 0)

m_rf = rf()
cross_val_score(m_rf, iris.data, iris.target, cv = m_kfold)
# array([0.9771134 , 0.9028    , 0.995     , 0.97869318, 0.8940246 ])
np.repeat(1,4)
#[교차검증 시각화]
v_mean = []
for i in range(1,4) : 
    m_rf = rf(max_features = i)
    v_score = cross_val_score(m_rf, iris.data, iris.target, cv = m_kfold)
    v_x = np.repeat(i,5)
    plt.scatter(v_x, v_score)
    v_mean.append(v_score.mean())
    
plt.ylim([0.8,1.1])
plt.plot(range(1,4), v_mean)
