# -------------------------------------------------
# @file  folo_proto1.py
# @brief とりあえずcifar10学習済みモデルへUSBカメラで
#        撮影した画像を送り、判定結果を画面に表示する.
# @note  .
# -------------------------------------------------

import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# define定義
# 入力画像の次元
img_column, img_row = 32, 32
# チャネル数（RGBなので3）
img_channels = 3
# クラス数
nclasses = 10
# クラスラベル
class_label_name = ["0_airplane", "1_car", "2_bird", "3_cat", "4_deer", "5_dog", "6_frog", "7_horse", "8_ship", "9_truck"]

# クラスラベルと学習済みモデルを保存するフォルダ
DATA_DIR = './'
# 学習済みモデルファイル
LEARNED_MODEL_FILE = 'cifar10.h5'

'''
LOAD MODEL AND PREDICT
'''
# 学習済みのモデルをロード
model = load_model(DATA_DIR+LEARNED_MODEL_FILE, compile=False)

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)
ret = cap.set(3,img_row)       # cam.CV_CAP_PROP_FRAME_WIDTH
ret = cap.set(4,img_column)    # cam.CAP_PROP_FRAME_HEIGHT

while True:
    # フレームをキャプチャする
    ret, frame = cap.read()
    # 画面に表示する
    cv2.imshow('frame',frame)
    # キーボード入力待ち
    key = cv2.waitKey(1) & 0xFF

#    if cv2.waitKey(1) & 0xFF == ord('c'):
    if key == ord('c'):
       img = np.array(frame, dtype=float) / 255.0
       # 各クラスの確率を計算する
       ret = model.predict(img, batch_size=1)   # OK

       # DataFrameを生成
       pd.options.display.float_format = '{:.3f}'.format # 小数点以下３桁までで切り捨てて表示
       df = pd.DataFrame(ret, columns=class_label_name)
       df['sum'] = df.sum(axis=1)
       print(df)

       # 評価対象画像を表示する
       plt.figure(1, figsize=(12, 3.2))
       plt.imshow(img)

       # 一番確率が高いクラスを選択する
       bestnum = 0.0
       bestclass = 0
       for n in range(nclasses):
           if bestnum < ret[0][n]:
               bestnum = ret[0][n]
               bestclass = n
       plt.text(2, -2, r'I think this picture is a ' + class_label_name[bestclass])
       print("I think this picture is a", class_label_name[bestclass])
       plt.show()

#    if cv2.waitKey(1) & 0xFF == ord('q'):
    if key == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

"""     # 各クラスの確率を計算する
    ret = model.predict(x_test[index:(index+1)], batch_size=1)   # OK

    # DataFrameを生成
    pd.options.display.float_format = '{:.3f}'.format # 小数点以下３桁までで切り捨てて表示
    df = pd.DataFrame(ret, columns=class_label_name)
    df['sum'] = df.sum(axis=1)
    print(df)

    # 評価対象画像を表示する
    plt.figure(1, figsize=(12, 3.2))
    plt.imshow(x_test[index])

    # 一番確率が高いクラスを選択する
    bestnum = 0.0
    bestclass = 0
    for n in range(nclasses):
        if bestnum < ret[0][n]:
            bestnum = ret[0][n]
            bestclass = n
    plt.text(2, -2, r'I think this picture is a ' + class_label_name[bestclass])
    print("I think this picture is a", class_label_name[bestclass])
    plt.show() """