#include <SPI.h>
#include <RF24.h>

#define PIN_CE    4
#define PIN_CSN   5
#define PIN_MOSI 23
#define PIN_MISO 19
#define PIN_SCK  18

RF24 radio(PIN_CE, PIN_CSN);
const byte ADDRESS[6] = "WBAN2";

struct __attribute__((packed)) PPGPacket {
  int32_t  ppg_raw;   // nilai PPG * 10000 (integer, dari Python)
  uint16_t seq;        // urutan paket, monoton, TIDAK direset tiap segmen
};
// sizeof(PPGPacket) = 4 + 2 = 6 byte

PPGPacket txPacket;
String serialBuffer = "";
#define LOG_INTERVAL 125

void setup() {
  Serial.begin(115200);
  delay(500);

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CSN);

  if (!radio.begin()) {
    Serial.println("[ERROR] nRF24L01 gagal");
    while (1);
  }

  radio.setPALevel(RF24_PA_HIGH);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(100);
  radio.enableDynamicPayloads();
  radio.openWritingPipe(ADDRESS);
  radio.stopListening();

  Serial.println("[OK] PPG TX READY");
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

  if (!raw.startsWith("START|") || !raw.endsWith("|END"))
    return;

  String ppgStr = extractField(raw, "PPG:");
  String seqStr = extractField(raw, "SEQ:");

  if (ppgStr == "" || seqStr == "") return;   // paket tidak lengkap, buang

  txPacket.ppg_raw = ppgStr.toInt();
  txPacket.seq      = (uint16_t) seqStr.toInt();

  bool ok = radio.write(&txPacket, sizeof(txPacket));

  if (txPacket.seq % LOG_INTERVAL == 0) {
    Serial.printf("[TX PPG] SEQ=%u RAW=%ld OK=%d\n",
                  txPacket.seq, (long)txPacket.ppg_raw, ok);
  }
}