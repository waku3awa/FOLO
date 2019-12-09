import serial
#from time import sleep

#シリアル通信(PC⇔Arduino)
ser = serial.Serial()
ser.port = "COM8"     #デバイスマネージャでArduinoのポート確認
ser.baudrate = 9600   #Arduinoと合わせる
ser.setDTR(False)     #DTRを常にLOWにしReset阻止
ser.open()            #COMポートを開く
ser.write(b'BACK')       #送りたい内容をバイト列で送信

#sleep(0.5)

line = ser.readline()  # 行終端'¥n'までリードする
print(line)

ser.close()           #COMポートを閉じる