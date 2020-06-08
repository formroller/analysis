import os 
os.getcwd()
os.chdir('')

run profile1

# 1. Data Loading
from sklearn.datasets import load_breast_cancer as cancer

df_cancer = cancer()
cancer_x = df_cancer.data
cancer_y = df_cancer.target

 # 2.scaling
from sklearn.preprocessing import StandardScaler
m_sc = StandardScaler()
m_sc.fit(cancer_x)
cancer_x = m_sc.transform(cancer_x)


 # 3. Y값의 전처리
import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder   # y값을 숫자로 변경하기 위해 사

cancer_y = np_utils.to_categorical(cancer_y)

# 4.train / test set 분리
from sklearn.model_selection import train_test_split
train_x , test_x, train_y, test_y = train_test_split(cancer_x, cancer_y, random_state = 0)


# [ 참고 : LabelEncoder예제 ] 
# Y값이 문자로 들어올 경우

s1 = Series(['a','b','c'])
f1 = LabelEncoder()
f1.fit(s1)
s2 = f1.transform(s1)

np_utils.to_categorical(s2)  # dummies 형태로 변경 (=get_dummies/pandas)

 # 5. ANN 모델 생성
import keras
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
model.add(Dense(15,
                input_dim = cancer_x.shape[1],
                activation= 'relu'))
model.add(Dense(2, activation = 'softmax')) # Dense(0,1)로 2개 값 입력 1,0 -> softmax
model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics = ['accuracy']  )

# [model compile]
# loss : 오차함수
 - 오차와 기울기 관계파악 위해사용
#optimizer : 속도 개선
# metrics : 평가 함수

model.fit(train_x, train_y, epochs= 50, batch_size=1)

model.evaluate(test_x, test_y)[0]   # Loss
model.evaluate(test_x, test_y)[1]   # accuracy

# 6. 모델 저장
 - 모델을 구성하는 각 뉴런의 가중치 저장
 - 모델 컴파일 요소 (Loss함수, 옵티마이저..) 저장
from keras.models import load_model
model.save('model_ann_cancer.hs')  # hs 확장자로 전달

dir(model)  # 하위 패키지 확인
model2 = load_model('model_ann_cancer.hs')

# (accuracy)
model.evaluate(test_x, test_y)[1]
model2.evaluate(test_x, test_y)[1]

model2.predict(test_x[0].reshape(1,30))

# 7. 모델 시각화
 # step1) 윈도우 graphviz 설치
  # https://graphviz.gitlab.io/_pages/Download/Download_windows.html
 
 # step2) 설치 후 위치 PATH 등록
  # 2-1)GRAPHVIZ_DOT 변수 생성
   # 위치\bin\dot.exe
   
  # 2-2) PATH추가
   # C:\Program Files(x86)\GRAPHVIZ2.38
   
 # SETP3) graphviz설치
 # pip install graphviz
 # conda install graphviz
from keras.utils import plot_model
import pydot
dir(pydot)
import graphviz

plot_model(model2, to_file ='ann_model2.png', show_shapes=True)

