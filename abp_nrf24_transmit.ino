#include <SPI.h>
#include <RF24.h>

// ─── PIN ───────────────────────────────────────────────────
#define PIN_CE    4
#define PIN_CSN   5
#define PIN_MOSI 23
#define PIN_MISO 19
#define PIN_SCK  18
// ───────────────────────────────────────────────────────────

RF24 radio(PIN_CE, PIN_CSN);

const byte ADDRESS[6] = "WBAN2";   // harus sama dengan RX

// ─── Struktur paket RF (26 bytes, max nRF24 = 32 bytes) ────
struct ABPPacket {
  int32_t  abp_raw;    // nilai ABP normalisasi ×1000 (dari field ABP:)
  float    abp_mmhg;   // tekanan darah asli dalam mmHg
  float    sbp;        // Systolic BP (mmHg)
  float    dbp;        // Diastolic BP (mmHg)
  float    map_val;    // Mean Arterial Pressure (mmHg)
  float    hr;         // Heart Rate (BPM) dari TX Python
  uint16_t seq;        // sequence number (mod 65535)
};
// ───────────────────────────────────────────────────────────

ABPPacket txPacket;
String    serialBuffer = "";

// Hitung berapa paket per detik untuk log (FS = 125 Hz)
#define LOG_INTERVAL 125


void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=============================================");
  Serial.println("  ESP32 ABP WBAN Transmitter");
  Serial.println("  Sumber: gabungan_v3_ABP.py → nRF24L01");
  Serial.println("=============================================");

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CSN);

  if (!radio.begin()) {
    Serial.println("[ERROR] nRF24L01 tidak terdeteksi!");
    Serial.println("        Periksa koneksi kabel.");
    while (true) delay(1000);
  }

  radio.setPALevel(RF24_PA_HIGH);     // HIGH untuk jangkauan WBAN
  radio.setDataRate(RF24_250KBPS);   // paling stabil untuk jarak dekat
  radio.setChannel(100);              // 2.476 GHz, hindari kanal WiFi
  radio.setPayloadSize(sizeof(ABPPacket));
  radio.openWritingPipe(ADDRESS);
  radio.stopListening();              // mode transmitter

  Serial.print("[OK] nRF24L01 siap! Payload size = ");
  Serial.print(sizeof(ABPPacket));
  Serial.println(" bytes");
  Serial.println("[..] Menunggu data ABP dari Python...\n");
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


// ─── Helper: ekstrak nilai string di antara key: dan | berikutnya ──
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

  // Validasi format paket
  // START|ABP:-123|MMHG:86.50|HR:72.0|SBP:120.0|DBP:80.0|MAP:93.3|SEQ:500|TS:...|END
  if (!raw.startsWith("START|") || !raw.endsWith("|END")) return;

  // Parse setiap field
  String abpStr  = extractField(raw, "ABP:");
  String mmhgStr = extractField(raw, "MMHG:");
  String hrStr   = extractField(raw, "HR:");
  String sbpStr  = extractField(raw, "SBP:");
  String dbpStr  = extractField(raw, "DBP:");
  String mapStr  = extractField(raw, "MAP:");
  String seqStr  = extractField(raw, "SEQ:");

  // Validasi semua field wajib tersedia
  if (abpStr.length() == 0 || seqStr.length() == 0) return;

  // Isi struct paket
  txPacket.abp_raw  = (int32_t) abpStr.toInt();
  txPacket.abp_mmhg = mmhgStr.toFloat();
  txPacket.sbp      = sbpStr.toFloat();
  txPacket.dbp      = dbpStr.toFloat();
  txPacket.map_val  = mapStr.toFloat();
  txPacket.hr       = hrStr.toFloat();
  txPacket.seq      = (uint16_t)(seqStr.toInt() % 65536);

  // Transmit via nRF24L01
  bool ok = radio.write(&txPacket, sizeof(txPacket));

  // Log tiap LOG_INTERVAL paket (≈ 1 detik pada FS=125 Hz)
  if (txPacket.seq % LOG_INTERVAL == 0) {
    if (ok) {
      Serial.printf("[TX OK  ] SEQ=%06d | ABP=%d | %.1f mmHg | HR=%.0f BPM\n",
                    txPacket.seq, txPacket.abp_raw,
                    txPacket.abp_mmhg, txPacket.hr);
    } else {
      Serial.printf("[TX FAIL] SEQ=%06d | cek antena / jarak\n",
                    txPacket.seq);
    }
  }
}