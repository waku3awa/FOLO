# -------------------------------------------------
# @file  folo_main.py
# @brief とりあえずcifar10学習済みモデルへUSBカメラで
#        撮影した画像を送り、判定結果を画面に表示する.
# @note  .
# -------------------------------------------------

import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# define定義 --------------------------------------
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

# 初期設定 ----------------------------------------
# 学習済みのモデルをロード
model = load_model(DATA_DIR+LEARNED_MODEL_FILE, compile=False)

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)
ret = cap.set(3,img_row)       # cam.CV_CAP_PROP_FRAME_WIDTH
ret = cap.set(4,img_column)    # cam.CAP_PROP_FRAME_HEIGHT
plt.ion()           # 対話モードオン

# -----------------------------------------------
while True:
    print ('コマンド c:キャプチャー、q:終了')
    key = input('>>>  ')
    if key == 'c':
        plt.close()
        # フレームをキャプチャする
        ret, frame = cap.read()
        height, width = frame.shape[:2]
        # キャプチャー画像を正方形にトリミングするために、
        # 縦横短い方に合わせてトリミング幅を取得する
        if width >= height:
            width_l = int((width - height) / 2)
            width_r = int(width - (width - height) / 2)
            height_t = 0
            height_b = height
        else:
            height_t = int((height - width) / 2)
            height_b = int(height - (height - width) / 2)
            width_l = 0
            width_r = width
        trim = frame[height_t:height_b, width_l:width_r]
        img = cv2.resize(trim, (img_column, img_row))
        img = np.array(img, dtype=float) / 255.0
        img = img[np.newaxis, :, :, :]  # 学習した時の配列構造に合わせるため

        # 各クラスの確率を計算する
        ret = model.predict(img, batch_size=1)
        # DataFrameを生成
        pd.options.display.float_format = '{:.3f}'.format # 小数点以下３桁までで切り捨てて表示
        df = pd.DataFrame(ret, columns=class_label_name)
        df['sum'] = df.sum(axis=1)
        print(df)       
        # 評価対象画像を表示する
        plt.figure(1, figsize=(2.5, 2))
        plt.imshow(img[0])
        # 一番確率が高いクラスを選択する
        bestnum = 0.0
        bestclass = 0
        for n in range(nclasses):
            if bestnum < ret[0][n]:
                bestnum = ret[0][n]
                bestclass = n
        plt.tick_params(labelbottom=False, labelleft=False,
                labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False,
                right=False, top=False)
        plt.text(-4, -2, 'I think this picture is a ' + class_label_name[bestclass])
        print("I think this picture is a", class_label_name[bestclass])
        plt.show()
        plt.pause(0.01)

    if key == 'q':
        break

# 終了処理 --------------------------------------
cap.release()
#cv2.destroyAllWindows()
plt.close()
