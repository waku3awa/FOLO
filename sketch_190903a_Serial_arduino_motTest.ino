//シリアル通信(PC⇔Arduino)

// 定義
/*#define STOP 0
#define FORWARD 1
#define BACK 2
#define RIGHT 3
#define LEFT 4
*/

// 変数
String data;
char led;

void setup() {
 Serial.begin(9600);
 pinMode(13, OUTPUT);
}

void loop(){
  if (Serial.available() > 0) {
    data = Serial.readString();
    if(data == "STOP"){
      Serial.println("FOLO IS STOPPING");
    }else if(data == "FORWARD"){
      Serial.println("FOLO IS GOING FORWARD.");
    }else if(data == "BACK"){
      Serial.println("FOLO IS GOING BACK.");
    }else if(data == "RIGHT"){
      Serial.println("FOLO IS TURNING RIGHT.");
    }else if(data == "LEFT"){
      Serial.println("FOLO IS TURNING LEFT.");
    }else{
      Serial.println("YOUR ORDER IS NOT AVAILABLE");
    }
  }else{
    ;
  }
}
