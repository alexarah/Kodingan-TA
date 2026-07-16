#include <SPI.h>
#include <RF24.h>

#define PIN_CE    4
#define PIN_CSN   5
#define PIN_MOSI 23
#define PIN_MISO 19
#define PIN_SCK  18

RF24 radio(PIN_CE, PIN_CSN);

const byte ADDRESS[6] = "WBAN3";  

struct PPGPacket {
  int32_t  ppg_raw;   // nilai PPG x10000 (untuk presisi 4 desimal)
  float    ppg_val;   // nilai PPG asli (0.0 – 4.0)
  uint16_t seq;
};
// 4 + 4 + 2 = 10 bytes

PPGPacket txPacket;
String    serialBuffer = "";

#define LOG_INTERVAL 125  // log tiap ~1 detik (FS=125 Hz)

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=============================================");
  Serial.println("  ESP32 PPG WBAN Transmitter");
  Serial.println("  Sumber: gabungan_PPG.py → nRF24L01");
  Serial.println("=============================================");

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CSN);

  if (!radio.begin()) {
    Serial.println("[ERROR] nRF24L01 tidak terdeteksi!");
    while (true) delay(1000);
  }

  radio.setPALevel(RF24_PA_HIGH);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(100);
  radio.setPayloadSize(sizeof(PPGPacket));
  radio.openWritingPipe(ADDRESS);
  radio.stopListening();

  Serial.print("[OK] nRF24L01 siap! Payload size = ");
  Serial.print(sizeof(PPGPacket));
  Serial.println(" bytes");
  Serial.println("[..] Menunggu data PPG dari Python...\n");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    serialBuffer += c;
    if (c == '\n') {
      parseAndTransmit(serialBuffer);
      serialBuffer = "";
    }
  }
}

String extractField(const String& raw, const String& key) {
  int start = raw.indexOf(key);
  if (start < 0) return "";
  start += key.length();
  int end = raw.indexOf('|', start);
  if (end < 0) end = raw.length();
  return raw.substring(start, end);
}

void parseAndTransmit(String raw) {
  raw.trim();
  if (!raw.startsWith("START|") || !raw.endsWith("|END")) return;

  String ppgStr = extractField(raw, "PPG:");
  String seqStr = extractField(raw, "SEQ:");
  if (ppgStr.length() == 0 || seqStr.length() == 0) return;

  txPacket.ppg_raw = (int32_t)ppgStr.toInt();
  txPacket.ppg_val = txPacket.ppg_raw / 10000.0f;
  txPacket.seq     = (uint16_t)(seqStr.toInt() % 65536);

  bool ok = radio.write(&txPacket, sizeof(txPacket));

  if (txPacket.seq % LOG_INTERVAL == 0) {
    if (ok) {
      Serial.printf("[TX OK  ] SEQ=%06d | PPG_raw=%d | %.4f\n",
                    txPacket.seq, txPacket.ppg_raw, txPacket.ppg_val);
    } else {
      Serial.printf("[TX FAIL] SEQ=%06d\n", txPacket.seq);
    }
  }
}