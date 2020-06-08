run profil1

# cancer data - ANN
# 1. Data loading
df_cacer = cancer()
cancer_x = df_cacer.data
cancer_y = df_cacer.target

# 2. scaling 
m_sc = StandardScaler()
m_sc.fit(cancer_x)
cancer_x = m_sc.transform(cancer_x)

# 3. Y값 전처리
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
cancer_y = np_utils.to_categorical(cancer_y)

# 4. train, test data 분리
train_x, test_x, train_y, test_y = train_test_split(cancer_x, 
                                                    cancer_y, 
                                                    random_state=0)

# [ 참고 : LabelEncoder 예 ]
s1 = Series(['a','b','c'])
f1 = LabelEncoder()
f1.fit(s1)
s2 = f1.transform(s1)

np_utils.to_categorical(s2)

# 5. ANN 모델 생성
import keras
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
model.add(Dense(15, 
                input_dim=cancer_x.shape[1],
                activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',    # 오차함수
              optimizer='adam',              # 속도개선
              metrics=['accuracy'])          # 평가함수

model.fit(train_x, train_y, epochs = 50, batch_size=1)
model.evaluate(test_x, test_y)[0]   # loss
model.evaluate(test_x, test_y)[1]   # accuracy

# 6. 모델 저장
# - 모델을 구성하는 각 뉴런의 가중치 저장
# - 모델의 컴파일 요소(loss함수, 옵티마이저..) 저장
from keras.models import load_model
model.save('model_ann_cancer.h5')   # h5  확장자로 저장

dir(model)
model2 = load_model('model_ann_cancer.h5')

model2.evaluate(test_x, test_y)[1]
model2.predict(test_x[0].reshape(1,30))

# 7. 모델 시각화
# step1. 윈도우용 graphviz 설치
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# step2. 설치 후 설치 위치 PATH 등록

# step 3. python graphviz 설치
# pip install graphviz
# conda install graphviz

from keras.utils import plot_model
plot_model(model, to_file='ann_model2.png', show_shapes=True)












