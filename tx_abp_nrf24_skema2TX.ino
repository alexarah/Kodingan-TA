#include <SPI.h>
#include <RF24.h>

#define PIN_CE    4
#define PIN_CSN   5
#define PIN_MOSI 23
#define PIN_MISO 19
#define PIN_SCK  18

RF24 radio(PIN_CE, PIN_CSN);
const byte ADDRESS[6] = "WBAN2";

struct __attribute__((packed)) ABPPacket {
  int32_t  abp_raw;   // nilai ABP * 100 (integer, dari Python)
  int16_t  sbp_x10;   // SBP * 10 (presisi 0.1 mmHg), hasil hitung di sisi TX
  int16_t  dbp_x10;   // DBP * 10 (presisi 0.1 mmHg), hasil hitung di sisi TX
  uint16_t seq;        // urutan paket
};
// sizeof(ABPPacket) = 4 + 2 + 2 + 2 = 10 byte

ABPPacket txPacket;
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

  Serial.println("[OK] ABP TX READY");
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

  String abpStr = extractField(raw, "ABP:");
  String sbpStr = extractField(raw, "SBP:");
  String dbpStr = extractField(raw, "DBP:");
  String seqStr = extractField(raw, "SEQ:");

  if (abpStr == "" || seqStr == "") return;   // paket tidak lengkap, buang

  txPacket.abp_raw = abpStr.toInt();
  txPacket.sbp_x10 = (int16_t) round(sbpStr.toFloat() * 10.0);
  txPacket.dbp_x10 = (int16_t) round(dbpStr.toFloat() * 10.0);
  txPacket.seq      = (uint16_t) seqStr.toInt();

  bool ok = radio.write(&txPacket, sizeof(txPacket));

  if (txPacket.seq % LOG_INTERVAL == 0) {
    Serial.printf("[TX ABP] SEQ=%u RAW=%ld SBP=%.1f DBP=%.1f OK=%d\n",
                  txPacket.seq, (long)txPacket.abp_raw,
                  txPacket.sbp_x10 / 10.0, txPacket.dbp_x10 / 10.0, ok);
  }
}
