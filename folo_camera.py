import cv2
# カメラのキャプチャを開始 --- (*1)
cam = cv2.VideoCapture(0)
ret = cam.set(3,320)
ret = cam.set(4,180)
#ret = cam.set(cam.CAP_PROP_CONVERT_RGB)
while True:
# フレームをキャプチャする
    ret, frame = cam.read()

    # 画面に表示する
    cv2.imshow('frame',frame)

    # キーボード入力待ち
    key = cv2.waitKey(1) & 0xFF

    # qが押された場合は終了する
    if key == ord('q'):
        break
    # sが押された場合は保存する
    if key == ord('s'):
        path = "photo" + img_num + ".jpg"
        cv2.imwrite(path, frame)

    # 画像を取得 --- (*2)
#    _, img1 = cam.read()
    
#   img1 = cam.read()
#   img2 = cam.imread("Lenna2.png",0)#グレースケールで読み込み
#    cv2.imwrite("Lenna1.png",img1)

#    imgC = cv2.imread("Lenna1.png")
#    cv2.imshow("color", imgC)

#    cam.imshow("gray", img2)
    # ウィンドウに画像を表示 --- (*3)
#    cv2.imshow('PUSH ENTER KEY', img)
    # Enterキーが押されたら終了する
#    if cv2.waitKey(1) == 13: break
# 後始末
cam.release()
cv2.destroyAllWindows()

