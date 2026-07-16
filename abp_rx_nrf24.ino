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

// ─── ADDRESS & STRUCT — harus identik dengan TX ────────────
const byte ADDRESS[6] = "WBAN2";

struct ABPPacket {
  int32_t  abp_raw;
  float    abp_mmhg;
  float    sbp;
  float    dbp;
  float    map_val;
  float    hr;
  uint16_t seq;
};

// ─── Variabel tracking metrik ──────────────────────────────
uint16_t seqTerakhir   = 0;
bool     pertama       = true;
uint32_t totalDiterima = 0;
uint32_t totalHilang   = 0;

#define LOG_INTERVAL 125   // log debug tiap ≈1 detik (FS=125 Hz)


void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=============================================");
  Serial.println("  ESP32 ABP WBAN Receiver");
  Serial.println("  nRF24L01 → Serial USB → gabungan_v3_ABP.py");
  Serial.println("=============================================");

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CSN);

  if (!radio.begin()) {
    Serial.println("[ERROR] nRF24L01 tidak terdeteksi!");
    Serial.println("        Periksa koneksi kabel.");
    while (true) delay(1000);
  }

  // ─── Konfigurasi HARUS sama persis dengan TX ───────────
  radio.setPALevel(RF24_PA_LOW);      // RX bisa LOW
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(100);
  radio.setPayloadSize(sizeof(ABPPacket));
  radio.openReadingPipe(0, ADDRESS);
  radio.startListening();             // mode receiver
  // ─────────────────────────────────────────────────────────

  Serial.print("[OK] nRF24L01 siap sebagai RECEIVER! Payload size = ");
  Serial.print(sizeof(ABPPacket));
  Serial.println(" bytes");
  Serial.println("[..] Menunggu paket ABP dari TX...\n");
}


void loop() {
  if (!radio.available()) return;

  ABPPacket pkt;
  radio.read(&pkt, sizeof(pkt));
  totalDiterima++;

  // ─── Hitung packet loss dari lompatan SEQ ──────────────
  uint16_t hilangFrame = 0;
  if (!pertama) {
    uint16_t expected = seqTerakhir + 1;
    if (pkt.seq != expected) {
      // Handle rollover uint16 (0–65535)
      hilangFrame = (uint16_t)(pkt.seq - expected);
      totalHilang += hilangFrame;
    }
  }
  pertama     = false;
  seqTerakhir = pkt.seq;

  // ─── kekuatan sinyal nRF24L01 ───────────────
  // 1 = sinyal di atas -64 dBm (kuat), 0 = lemah
  bool rpd = radio.testRPD();

  // ─── Packet loss rate ───────────────────────────────────
  uint32_t totalExpected = totalDiterima + totalHilang;
  float lossPct = (totalExpected > 0)
    ? (float)totalHilang / totalExpected * 100.0f
    : 0.0f;

  // ─── Timestamp sisi RX (ms sejak boot) ─────────────────
  unsigned long ts_rx = millis();

  // ─── Kirim ke Python dalam format yang diharapkan thread_rx() ──
  // Format: START|ABP:{raw}|MMHG:{mmhg}|HR:{hr}|SBP:{sbp}|DBP:{dbp}|MAP:{map}|SEQ:{seq}|TS:{ts}|END
  Serial.print("START|ABP:");
  Serial.print(pkt.abp_raw);
  Serial.print("|MMHG:");
  Serial.print(pkt.abp_mmhg, 2);
  Serial.print("|HR:");
  Serial.print(pkt.hr, 1);
  Serial.print("|SBP:");
  Serial.print(pkt.sbp, 1);
  Serial.print("|DBP:");
  Serial.print(pkt.dbp, 1);
  Serial.print("|MAP:");
  Serial.print(pkt.map_val, 1);
  Serial.print("|SEQ:");
  Serial.print(pkt.seq);
  Serial.print("|TS:");
  Serial.print(ts_rx);
  Serial.println("|END");

}
