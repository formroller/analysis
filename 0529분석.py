run profile1
df1 = pd.read_csv('ThoraricSurgery.csv', header = None)

df1_x = df1.iloc[:,1:17]  # df1.drop(0,axis=1)
df1_y = df1.iloc[:,17]

train_x, test_x, train_y, test_y=train_test_split(df1_x,
                                                  df1_y,
                                                  random_state = 0)


m_svc = SVC()
m_svc.fit(train_x, train_y)
m_svc.score(test_x, test_y) # 90.68%

# fit시 데이터 프레임 아닌 어레이 요구할 경우 .values
# train_x.values / test_x.values  *array 형식으로 출력

# interaction / scalling / PCA ....

# =============================================================================
# (PCA + KNN) 이미지 인식 / 분석
# =============================================================================
#sklearn에서의 이미지 데이터 셋
- 2000년 초반 이후 유명인사 얼굴 데이터
- 전처리 속도를 위해 흑백으로 제공
- 총 62명 얼굴을 여러장 활용한 데이터 셋
- 총 32023개의 이미지 데이터, 87 * 65 픽셀로 규격화

 - 3*3 형식 -> 1*9 형식 flat 비교
 - 비정형 데이터 => 전처리 어려움
 - 픽셀이 모두 설명변수이다.**
 - 픽셀별 의미있는 가중치 부여해 분류(예측)함에 있어 주요 변수를 도출
 - 픽셀별 각각의 중요도 파악 
 => 차원 유지하며 인근 픽셀의 특징(다름의 정도가)비교 (convolution, 합성곱) - cnn
 - 규격화 => 해상도 다를 경우 비교 설명 변수 갯수 달라짐. 비교분석 불가
 
# [KNN 모델 - 거리기반]
 => 픽셀간 차이로 분류하는 것이 효과적 
 => RGB간 거리에 따라 분류
 => 가중치가 동일
# [PCA]
 - 모든 설명변수 고려한 새로운 인공변수 유도
 - 모든 설명변수 결합


# 1. data set 추출
from sklearn.datasets import fetch_lfw_people

# 다운로드
people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# data set 구성
people.keys()
people.images.shape  # (2749, 87, 65) => 실제 raw데이터
people.data.shape    # (2749, 5655) 층, 87*65 => 학습시킬 데이터

people.images[0]  # 첫번째 층 선택 - 첫 이미지 선택 => RGB값 출력

fig, ax = plt.subplots(1,3, figsize = (15,0))  # 1차원

#[subplot 1차원]

ax[0].imshow(people.images[2])
ax[1].imshow(people.images[2348])  
ax[2].imshow(people.images[300])
# imshow, RGB값을 갖는 로우 데이터로부터 실제 이미지로 출력하는 시각화 함수

# =============================================================================
#[ 참고 - 2차원 형식의 subplot을 1차원으로 평탄화시켜 순차적으로 시각화]

fig, ax1 = plt.subplots(2,3, figsize = (15,0))  # 2차원
ax1[0,0].imshow(people.images[185])  # 2차원 형식으로 축 지정 필요
ax1[0,1].imshow(people.images[4])

ax1 = ax1.ravel()
ax1[4].imshow(people.images[343]) # 1차원 형식으로 축 전달 가능
a1 = np.arange(6).reshape(2,3)


a1.flatten()  # 평탄화 함수 / 딥카피 유무 차이
a1.ravel()    # 평탄화 함수 / 
#=> 하나의 for문만 사용하기 위해 평탄화 작업 실시
# =============================================================================

people.target[300]       # 이름을 숫자로 변경한 값
people.target_names[50]  # 실제 이름

name1 = people.target_names[people.target[2]]
name2 = people.target_names[people.target[2348]]
name3 = people.target_names[people.target[300]]

ax[0].set_title(name1)
ax[1].set_title(name2)
ax[2].set_title(name3)

# 2. Data Sampling(사람별 최대 50개 이미지 선택)

80 <- 20 upsampling  (대 <- 소), 데이터 확보가 목적(대용량 데이터에서는 효과 좋지 않음)
80 -> 20 downsampling(대 -> 소)
#enumerate - 숫자(위치값) 부여 함수
np.bincount(people.target) # 타겟이 나온 횟수 계산
# 2-1)downsampling 
np.where(people.target == 13) # target이 13인 데이터의 위치값 추출 (색인)
                              # (참/거짓 미전달시)조건에 매칭되는 위치값 출력
np.where(people.target == 13)[0][:50]  # 50개의 array 생성

# 위치값 출력)
v_row=[]
for i in np.unique(people.target) : 
    v_row = v_row + list(np.where(people.target == i)[0][:50])
v_row

x_people = people.data[v_row]
y_people = people.target[v_row]
## 다차원 다운샘플링 코드 학습 필요
# =============================================================================
# 참고, 위 코드 상세
li=[]
li.append(list(np.where(people.target == 13)[0][:50])) # list에 list삽입하는 방식(추천X)
li + list(np.where(people.target == 13)[0][:50])  # 1차원 list로 삽입.
# =============================================================================

# 3. scaling 
people.data[0].shape # 설명변수 - RGB값
people.data[0].max() # RGB값 -> 0 ~ 255 범위 (따라서 255로 나누어 스케일링)
x_people_sc = x_people / 255  # MinMaxScaler() 방식과 동일

# 4. train, test set 분리
train_x, test_x, train_y, test_y = train_test_split(x_people_sc,
                                                    y_people,
                                                    random_state=0)
# 5. k-nn 학습
m_knn = knn(n_neighbors = 3)
m_knn.fit(train_x, train_y)

# 6. k-nn모델 평가
m_knn.score(test_x, test_y)  # 20.98%  PCA이전 결과값(확보된 train data 표본 갯수가 적다.)

# 7. PCA기법을 통한 인공변수 유도 (변수 가공) * 100 -> 100차원
  # - PCA -> 이미지 분석의 경우 의미있는 변수의 유도 목적.(픽셀간 결합)
m_pca = PCA(n_components=100, whiten = True)  # whiten = True ,표준화수행
m_pca.fit(train_x)

train_x_pca = m_pca.transform(train_x)
test_x_pca = m_pca.transform(test_x)

# 8. 인공변수(100개)로 knn모델 학습 및 평가
m_knn = knn(n_neighbors = 3)
m_knn.fit(train_x_pca,train_y)
m_knn.score(test_x_pca, test_y)  # 22.77%
                                 # 28.8% whiten 이후 예측력 향상

# =============================================================================
# CV 기법 (Cross Validation - 교차검증)
# =============================================================================
[목적]
 - 평가 점수의 일반화
 - train, test data set 분리시 더 
  1) 좋은 데이터로 추출 및 높은 점수 나올 사능성
  2) 안좋은 데이터 추출, 낮은 점수가 나올 가능성
     => 위에대해 충분히 테스트 일반화한다.
 - k-fold 기법은 전체 데이터를 k등분해 k개의 test set에 대해 평가하는 형식
   
 => 전체 데이터 테스트와 중복되지 않은 데이터로 테스트한 결과를 갖는다
 
 # k-fold - iris data
#1) 데이터 로딩
df_iris = iris()
 
#2) k-fold 수행

#CV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

m_kfold = KFold(n_splits = 5,     #  k의 수 (교차검증 횟수)
                shuffle = True,   # 데이터의 shuffle 여부
                random_state=0)   # seed 값

cross_val_score(estimator,   # 적용하고자 하는 모델
                x,           # 학습 데이터의 설명변수 set
                y,           # 학습 데이터의 종속변수 set
                cv)          # 교차검증 방법

m_rf = rf()
cross_val_score(m_rf, df_iris.data, df_iris.target, cv=m_kfold)
v_score = cross_val_score(m_rf, df_iris.data, df_iris.target, cv=m_kfold).mean()  # 94.44% - 평균값(일반화된 점수)

# 교차검증 과정 시각화
v_mean = []

for i in range(1,4) :
    m_rf = rf(max_features = i)
    v_score = cross_val_score(m_rf, df_iris.data, df_iris.target, cv = m_kfold)
    v_x = np.repeat(i,5)
    plt.scatter(v_x,v_score)
    v_mean.append(v_score.mean())

plt.ylim([0.8,1.1])
plt.plot(range(1,4), v_mean)

# [ 참고 - RGB값 추출 ]
import imageio
im = imageio.imread('cat1.jpg')
im.shape
imageio.imwrite('cat11.jpg', im[:, :, 0])

im2 = imageio.imread('cat11.jpg')
im2.shape






