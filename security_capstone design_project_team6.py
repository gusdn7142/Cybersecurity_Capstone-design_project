import tensorflow as tf
import numpy as np                                       #numpy모듈의 이름을 np로 지정
from tensorflow import keras                             #tensorflow모듈에서 keras 가져오기
import matplotlib.pyplot as plt                          #matplotlib모듈 이름을 plt로 지정
from keras.preprocessing.image import ImageDataGenerator #keras모듈의 ImageDataGenerator import


# 6계층 CNN모델(He방식으로 가중치 초기화)
# ananconda 가상환경 + pycharm IDE + tensorflow(반드시 1.14버전), pillow, keras, matplot 라이브러리 등을 사용함
# 데이터셋을 바탕화면에 풀면 된다.

seed = 1                 #랜덤시드를 정함
np.random.seed(seed)
tf.set_random_seed(seed)


# 2. 학습용 이미지 데이터 불러오기

from tensorflow import keras  # tensorflow모듈에서 keras 사용
import matplotlib.pyplot as plt  # matplot모듈 import
from keras.preprocessing.image import ImageDataGenerator  # kreas모듈에서 imageDataGenerator import

data_generator = ImageDataGenerator(rescale=1. / 255)  # 픽셀값은 0~255값을 0~1로 데이터 전처리
train_generator = data_generator.flow_from_directory('C:/Users/samsung/Desktop/train',  # 0라벨과 1라벨 이미지 총 380개를 불러온다.
                                                     target_size=(200, 200),  # 이미지 크기
                                                     color_mode='grayscale',  # 이미지 색상
                                                     shuffle=True,  # 이미지를 랜덤으로 불러온다.
                                                     batch_size=19,  # 이미지 380개를 한번 불러올떄 55개로 나누어 불러온다.
                                                     class_mode='binary')  # 라벨 분류방식을 binary으로 설정

x = tf.placeholder("float", shape=[None, 40000])  # 이미지를 저장할 텐서변수 선언
x_image = tf.reshape(x, [-1, 200, 200, 1])  # 이미지의 shape를 모델의 input에 맞게 변경

y = tf.placeholder("float", shape=[None, 1])  # 라벨을 저장할 텐서변수 선언

"""모델 구현"""


# 편향 함수 정의
def bias_variable(shape):  # 0.1로 초기값 지정하여 원하는 사이즈로 리턴하는 함수
    initial = tf.constant(0.1, shape=shape)  # b(bias)를 0.1를 갖도록 초기화
    return tf.Variable(initial)


# 합성곱 계층 함수
def conv2d(x, w):  # convolution계층 정의 함수
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1],
                        padding='SAME')  # 스트라이드는 1로 지정하고 입력이미지크기와 출력이미지 크기는 padding을 SAME으로 하여 동일하게 만들어준다.


# 풀링 계층 함수
def max_pool_2x2(x):  # max-pooling계층 정의 함수
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')  # 풀링 크기는 2*2크기로 묶어서 풀링, 스트라이드는 2로 지정한다.


######CNN model layer_1#######       #첫번째 합성곱 계층(합성곱계층과 맥스풀링계층으로 구성)

w_conv1 = tf.get_variable("L1", shape=[3, 3, 1, 32],
                          initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬의 필터크기는 3*3, 필터수는 32개로 하고 초기화 방식으로는 He방법을 적용
b_conv1 = bias_variable([32])  # 32개 가중치행렬에 대한 bias

h_conv1 = tf.nn.relu(conv2d(x_image,
                            w_conv1) + b_conv1)  # 첫번째 합성곱 입력계층으로서, 입력 이미지 x_image 에 대해 convolution을 적용하고 결과를 2D 텐서 W_conv1에 리턴하고 b_conv1를 더한다
h_pool1 = max_pool_2x2(h_conv1)  # 첫번째 풀링계층으로서, max-pooling과정을 거쳐 특징맵(feature map) 영역에서 최대값을 가져온다,

print(h_conv1)  # 첫번째 합성곱계층 구조 출력
print(h_pool1)  # 첫번째 풀링계층 구조 출력

#######CNN model layer_2#######         #두번째 합성곱 계층(합성곱계층과 맥스풀링계층으로 구성)
w_conv2 = tf.get_variable("L2", shape=[3, 3, 32, 64],
                          initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬의 필터크기는 3*3, 필터수는 64개로 하고 초기화 방식으로는 He방법을 적용
b_conv2 = bias_variable([64])  # 64개 가중치행렬에 대한 bias

h_conv2 = tf.nn.relu(conv2d(h_pool1,
                            w_conv2) + b_conv2)  # 두번째 합성곱 계층으로서, 입력 특징맵(feature map) h_pool1에 대해 convolution을 적용하고 결과를 2D 텐서 W_conv2에 리턴하고 b_conv2 더한다
h_pool2 = max_pool_2x2(h_conv2)  # 두번째 풀링계층으로서,  max-pooling과정을 거쳐 특징맵(feature map) 영역에서 최대값을 가져온다,

print(h_conv2)  # 두번쨰 합성곱계층 구조 출력
print(h_pool2)  # 두번째 풀링계층 구조 출력

#######CNN model layer_3#######         #세번째 합성곱 계층(합성곱계층과 맥스풀링계층으로 구성)
w_conv3 = tf.get_variable("L3", shape=[3, 3, 64, 96],
                          initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬의 필터크기는 3*3, 필터수는 96개로 하고 초기화 방식으로는 He방법을 적용
b_conv3 = bias_variable([96])  # 96개 가중치행렬에 대한 bias

h_conv3 = tf.nn.relu(conv2d(h_pool2,
                            w_conv3) + b_conv3)  # 세번째 합성곱 계층으로서, 입력 특징맵(feature map) h_pool1에 대해 convolution을 적용하고 결과를 2D 텐서 W_conv2에 리턴하고 b_conv2 더한다
h_pool3 = max_pool_2x2(h_conv3)  # 세번째 풀링계층으로서, max-pooling과정을 거쳐 특징맵(feature map) 영역에서 최대값을 가져온다,

print(h_conv3)  # 세번째 합성곱계층 구조 출력
print(h_pool3)  # 세번째 풀링계층 구조 출력

#######CNN model layer_4#######     #네번째 합성곱 계층(합성곱계층과 맥스풀링계층으로 구성)
w_conv4 = tf.get_variable("L4", shape=[3, 3, 96, 96],
                          initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬의 필터크기는 3*3, 필터수는 96개로 하고 초기화 방식으로는 He방법을 적용
b_conv4 = bias_variable([96])  # 96개 가중치행렬에 대한 bias

h_conv4 = tf.nn.relu(conv2d(h_pool3,
                            w_conv4) + b_conv4)  # 네번째 합성곱 계층으로서, 입력 특징맵(feature map) h_pool1에 대해 convolution을 적용하고 결과를 2D 텐서 W_conv2에 리턴하고 b_conv2 더한다
h_pool4 = max_pool_2x2(h_conv4)  # 네번째 풀링계층으로서, max-pooling과정을 거쳐 특징맵(feature map) 영역에서 최대값을 가져온다,

print(h_conv4)  # 네번째 합성곱계층 구조 출력
print(h_pool4)  # 네번째 풀링계층 구조 출력

#######CNN model layer_5#######    #다섯번째 합성곱 계층(합성곱계층과 맥스풀링계층으로 구성)
w_conv5 = tf.get_variable("L5", shape=[3, 3, 96, 128],
                          initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬의 필터크기는 3*3, 필터수는 128개로 하고 초기화 방식으로는 He방법을 적용
b_conv5 = bias_variable([128])  # 128개 가중치행렬에 대한 bias

h_conv5 = tf.nn.relu(conv2d(h_pool4,
                            w_conv5) + b_conv5)  # 다섯번째 합성곱 계층으로서, 입력 특징맵(feature map) h_pool1에 대해 convolution을 적용하고 결과를 2D 텐서 W_conv2에 리턴하고 b_conv2 더한다
h_pool5 = max_pool_2x2(h_conv5)  # 다섯번째 풀링계층으로서, max-pooling과정을 거쳐 특징맵(feature map) 영역에서 최대값을 가져온다,

print(h_conv5)  # 다섯번째 합성곱계층 구조 출력
print(h_pool5)  # 다섯번째 풀링계층 구조 출력

#######CNN model layer_6#######     #여섯섯번째 합성곱 계층(합성곱계층과 맥스풀링계층으로 구성)
w_conv6 = tf.get_variable("L6", shape=[3, 3, 128, 128],
                          initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬의 필터크기는 3*3, 필터수는 128개로 하고 초기화 방식으로는 He방법을 적용
b_conv6 = bias_variable([128])  # 128개 가중치행렬에 대한 bias

h_conv6 = tf.nn.relu(conv2d(h_pool5,
                            w_conv6) + b_conv6)  # 여섯번째 합성곱 계층으로서, 입력 특징맵(feature map) h_pool1에 대해 convolution을 적용하고 결과를 2D 텐서 W_conv2에 리턴하고 b_conv2 더한다
h_pool6 = max_pool_2x2(h_conv6)  # 여섯번째 풀링계층으로서, max-pooling과정을 거쳐 특징맵(feature map) 영역에서 최대값을 가져온다,

print(h_conv6)  # 여섯번째 합성곱계층 구조 출력
print(h_pool6)  # 여섯번째 풀링계층 구조 출력

#######fully connected layer#######    #완전연결 계층
w_fc1 = tf.get_variable("fc1", shape=[4 * 4 * 128, 128],
                        initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬을 [높이*너비*채널,출력]으로 선언
b_fc1 = bias_variable([128])  # 128개 뉴런에 대한 bias

h_pool2_flat = tf.reshape(h_pool6, [-1, 4 * 4 * 128])  # sigmoid(시그모이드) 계층에 넣기위해 직렬화하기 위한 텐서,  합성곱계층에서 나온결과를 2차원 텐서로 변환
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)  # sigmoid(시그모이드) 계층에 넣기 위해 직렬화,  (일차원벡터 * 가중치행렬 + 편향)

print(h_fc1)  # 완전연결계층 구조 출력

#######drop out#######                  #드롭 아웃 계층
keep_prob = tf.placeholder("float")  # 뉴런이 drop out되지 않을 확률 저장
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 드롭아웃 실행,  뉴런의 출력을 자동으로 스케일링 해주는 함수

# 라벨 확률 출력 계층
w_fc2 = tf.get_variable("fc2", shape=[128, 1],
                        initializer=tf.contrib.layers.variance_scaling_initializer())  # 가중치 행렬의 뉴런수를 128개로 하고 초기화 방식으로는 He방법을 적용
b_fc2 = bias_variable([1])  # sigmoid(시그모이드)계층에 대한  bias

y_conv = tf.sigmoid(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)  #sigmoid(시그모이드)활성화 함수를 적용해  0~1사이의 최종결과값을 가진다. (0에 가까우면 0라벨과 유사하고 1에 가까우면 1라벨과 유사하다)

print(y_conv)  # 0~1사이의 값으로 출력

cost = -tf.reduce_mean(y * tf.log(y_conv) + (1 - y) * tf.log(1 - y_conv))  #비용함수 정의
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)                  #Adam최적화 알고리즘 적용(학습율은 0.0001로 적용)

predicted = tf.cast(y_conv > 0.5, "float")                            # 예측값이 0.5이상이면 1(=true), 아니면 0(=false)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), "float"))  # 예측 라벨값과 실제라벨값을 비교후 부울값을 실수로 변환후 정확도 계산

"""세션생성 및 초기화"""
sess = tf.Session()                          #세션 생성
sess.run(tf.global_variables_initializer())  #변수 초기화 및 세션 실행

grape_cost = []            #cost 시각화 변수
grape_accuracy = []        #accuarcy 시각화 변수

batch_accuracy = 0         #배치마다 accuarcy의 총합
step_average_accuarcy = 0  #스텝마다 평균 accuarcy

batch_cost = 0             # 배치마다 cost 총합
step_average_cost = 0      # 스텝마다 평균 cost

"""모델 학습"""
batch = 20  # 배치 크기
epoch = 15  # 전체 반복횟수

for step in range(epoch):  # 200개의 이미지 총 50번 학습

    for i in range(batch):                # 이미지 200개를 50개씩 iteration 4번으로 나눠서 학습
        total_x, total_y = train_generator.next()  # batch_size만큼 데이터셋 랜덤으로 불러오기
        total_x = np.reshape(total_x, (19, 40000))  # 60*60크기의 데이터셋 불러오기
        total_y = np.reshape(total_y, (19, 1))  # 라벨 0,1 shape 변형

        # 학습시키기
        sess.run(optimizer, feed_dict={x: total_x, y: total_y, keep_prob: 0.7})  # 학습시키기                                   #선19 개의 입력이미지에 대해 경사하강법 수행, 드롭아웃 안될확률 0.7

        # 손실율 불러옴
        cost_val = sess.run(cost, feed_dict={x: total_x, y: total_y, keep_prob: 0.7})  # 손실율 계산
        grape_cost.append(cost_val)                                                    #손실율 배열에 저장

        # 학습 정확도 불러옴
        train_accuracy = sess.run(accuracy, feed_dict={x: total_x, y: total_y, keep_prob: 0.7})   #배치마다 학습정확도 불러옴
        grape_accuracy.append(train_accuracy)

        # if i%1 == 0:                             #step마다 정확도와 손실율 측정
        print("%d번 반복중, 배치:(%d/%d), 학습 정확도: %g" % (step + 1, i + 1, batch, train_accuracy), " /  손실율:",
              cost_val)  # 배치마다 훈련데이터에 대한 정확도와 손실율 출력

        # 배치마다 학습정확도와 손실율 누적
        batch_accuracy += train_accuracy
        batch_cost += cost_val

        print("####################################################################")

        ####################################################################
        # 2. 테스트용 이미지 데이터 불러오기
        data_generator2 = ImageDataGenerator(rescale=1. / 255)  # 픽셀값을 0~255에서 0~1로 데이터 전처리
        test_generator = data_generator2.flow_from_directory('C:/Users/samsung/Desktop/test',  # 폴더명이 이미지들의 라벨이 됨
                                                             target_size=(200, 200),  #이미지 크기
                                                             color_mode='grayscale',  #이미지 색상
                                                             shuffle=False,           #이미지를 랜덤으로 불러온다.
                                                             batch_size=20,           # 이미지 20개를 한번에 불러온다.
                                                             class_mode='binary')     #라벨 분류방식을 binary으로 설정

        total_x2, total_y2 = test_generator.next()       # batch_size만큼 데이터셋 랜덤으로 불러오기
        total_x2 = np.reshape(total_x2, (20, 1, 40000))  # 모델 input에 맞게 이미지 shape 변형

        img1 = ["박1", "박2", "박3", "박4", "박5", "박6", "박7", "박8", "박9", "박10"]

        for m in range(0, 10):
            test_accuarcy1 = sess.run(y_conv, feed_dict={x: total_x2[m], keep_prob: 1.0})
            print("%s의 test 정확도 :" % img1[m], test_accuarcy1[0][0])                       #원본과 다른 이미지에 대한 정확도 출력
        print("#####################################################")

        img2 = ["copy1", "copy2", "copy3", "copy4", "copy5", "copy6", "copy7", "copy8", "copy9", "copy10"]

        count_b = 0
        for b in range(10, 20):
            test_accuarcy2 = sess.run(y_conv, feed_dict={x: total_x2[b], keep_prob: 1.0})
            print("%s의 test 정확도 :" % img2[count_b], test_accuarcy2[0][0])                 #원본과 유사한 이미지에 대한 정확도 출력
            count_b += 1
        print("#####################################################")

        """
        # 테스트셋 이미지 화면으로 확인하기
        import matplotlib.pyplot as plt

        fig = plt.figure()
        rows = 4
        cols = 5

        for k in range(0, 20):
            output1 = fig.add_subplot(rows, cols, k + 1)  # i+1
            output1.imshow(total_x2[k].reshape(200, 200), cmap="Greys", interpolation='nearest')
            # output1.set_title('1번째 image')
            # plt.tight_layout()     #자동으로 이미지 겹침 방지
        plt.show()
        """

        ####################################################################
        print(sep="\n")

    #반복마다 학습 정확도, 손실율 파악
    step_average_accuarcy = batch_accuracy / batch
    step_average_cost = batch_cost / batch

    print("%d번 반복후 평균 학습 정확도: %g" % (step + 1, step_average_accuarcy))  # 1회 반복마다 훈련데이터에 대한 평균 정확도 측정
    print("         평균 손실율:", step_average_cost)                            # 1회 반복마다 훈련데이터에 대한 평균 손실율 측정

    batch_cost = 0      #반복 1번 종료시 0으로 바꿔줌
    batch_accuracy = 0  #반복 1번 종료시 0으로 바꿔줌

    print("####################################################################")
    ####################################################################

    print(sep="\n")

h, c = sess.run([y_conv, predicted], feed_dict={x: total_x, y: total_y, keep_prob: 1.0})    #마지막 반복시 학습데이터에 대한 예측확률과 예측라벨 파악

print(sep="\n")

"""예측 확률과 예측라벨"""
#################################
print("-예측 확률")
for i in range(0, 10):
    print("확률(%):", h[i], end=', ')
print('')
for i in range(10, 19):
    print("확률(%):", h[i], end=', ')
print('')

print(sep="\n")

print("-예측 레이블")
for i in range(0, 10):
    print("label:", c[i], end=', ')
print('')
for i in range(10, 19):
    print("label:", c[i], end=', ')
print('')
#################################


print(sep="\n")

#################################
print("-실제 레이블")
for i in range(0, 10):
    print("label:", total_y[i], end=', ')
print('')
for i in range(10, 19):
    print("label:", total_y[i], end=', ')
print('')
############################

print(sep="\n")




#학습 정확도 시각화

epoch_count = range(1, len(grape_accuracy) + 1)

plt.subplot(2, 1, 1)                      # 그래프 2개중 1번으로 띄우기
plt.plot(epoch_count, grape_accuracy)
plt.title('model accuarcy')
plt.xlabel('epoch')
plt.ylabel('accuarcy')
plt.legend(['train'], loc='upper left')  # 그래프에 이름표시
# plt.show()


#학습 손실율 시각화
epoch_count = range(1, len(grape_cost) + 1)

plt.subplot(2, 1, 2)               # 그래프 2개중 2번으로 띄우기
plt.plot(epoch_count, grape_cost)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train'], loc='upper left')  # 그래프에 이름표시

plt.tight_layout()  # 자동으로 겹침 방지
plt.show()


