/*
  rx_abp_ppg_nrf24.ino
  =====================
  RX Side — ESP32 + nRF24L01

  Cocok untuk: rx_abp_ppg.py (script RX terpisah, tidak lagi satu
  proses dengan TX).

  Menerima:
    - ABP packet dari tx_abp_nrf24.ino (10 byte, packed)
    - PPG packet dari tx_ppg_nrf24.ino (6 byte, packed)
  dibedakan lewat ukuran payload dinamis (radio.getDynamicPayloadSize()).

  Forward ke Python via Serial USB, format PERSIS seperti yang
  diharapkan rx_abp_ppg.py:

    ABP:
    START|TYPE:ABP|ABP:<raw>|SBP:<sbp>|DBP:<dbp>|SEQ:<seq>|END

    PPG:
    START|TYPE:PPG|PPG:<raw>|SEQ:<seq>|END

  Catatan: TIDAK ada lagi field SEG/MMHG/HR/MAP/TS di baris ini —
  segmen aktif & nilai mmHg sekarang dihitung ulang di sisi Python
  dari SEQ (SEQ = urutan paket monoton per sinyal, PAKET_PER_SEGMEN
  di rx_abp_ppg.py harus sama dengan yang dipakai TX Python).

  SBP/DBP di radio dikirim sebagai int16 x10 (presisi 0.1 mmHg), jadi
  di sini dibagi 10.0 lagi sebelum diteruskan sebagai teks desimal ke
  Python — dari sisi Python format & presisinya SAMA PERSIS seperti
  sebelumnya, tidak ada yang perlu diubah di rx_abp_ppg.py.

  PENTING — konsistensi radio (harus SAMA di TX-ABP, TX-PPG, dan RX):
    - Channel        : 100
    - Data rate       : RF24_250KBPS
    - Address         : "WBAN2"
    - Dynamic payload : enableDynamicPayloads() (BUKAN setPayloadSize)
    - Struct ABPPacket/PPGPacket packed, urutan & tipe field sama
      persis dengan kedua file TX.
*/

#include <SPI.h>
#include <RF24.h>

#define PIN_CE    4
#define PIN_CSN   5
#define PIN_MOSI 23
#define PIN_MISO 19
#define PIN_SCK  18

RF24 radio(PIN_CE, PIN_CSN);
const byte ADDRESS[6] = "WBAN2";

// ─── Struct ABP (harus sama persis dengan tx_abp_nrf24.ino) ─
struct __attribute__((packed)) ABPPacket {
  int32_t  abp_raw;
  int16_t  sbp_x10;   // SBP * 10 (presisi 0.1 mmHg)
  int16_t  dbp_x10;   // DBP * 10 (presisi 0.1 mmHg)
  uint16_t seq;
};
// sizeof(ABPPacket) = 4 + 2 + 2 + 2 = 10 byte

// ─── Struct PPG (harus sama persis dengan tx_ppg_nrf24.ino) ─
struct __attribute__((packed)) PPGPacket {
  int32_t  ppg_raw;
  uint16_t seq;
};
// sizeof(PPGPacket) = 6 byte

uint32_t totalABP = 0;
uint32_t totalPPG = 0;

uint16_t lastSeqABP = 0;
uint16_t lastSeqPPG = 0;

uint32_t lostABP = 0;
uint32_t lostPPG = 0;

bool firstABP = true;
bool firstPPG = true;

#define LOG_INTERVAL 125

uint8_t buffer[32];


void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("=======================================");
  Serial.println(" ESP32 Combined WBAN Receiver");
  Serial.println("=======================================");

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CSN);

  if (!radio.begin()) {
    Serial.println("[ERROR] nRF24L01 gagal");
    while (1);
  }

  radio.setPALevel(RF24_PA_LOW);
  radio.setDataRate(RF24_250KBPS);
  radio.setChannel(100);
  radio.enableDynamicPayloads();   // wajib sama dengan kedua sisi TX
  radio.openReadingPipe(0, ADDRESS);
  radio.startListening();

  Serial.println("[OK] Receiver Ready");
}


void loop() {
  if (!radio.available()) return;

  uint8_t payloadSize = radio.getDynamicPayloadSize();

  if (payloadSize == 0 || payloadSize > 32) {
    radio.flush_rx();
    return;
  }

  radio.read(&buffer, payloadSize);

  if (payloadSize == sizeof(ABPPacket)) {
    handleABP();
  }
  else if (payloadSize == sizeof(PPGPacket)) {
    handlePPG();
  }
  // payload dengan ukuran lain diabaikan (kemungkinan noise/collision)
}


void handleABP() {
  ABPPacket* pkt = (ABPPacket*) buffer;
  totalABP++;

  if (!firstABP) {
    uint16_t expected = lastSeqABP + 1;
    if (pkt->seq != expected)
      lostABP += (uint16_t)(pkt->seq - expected);
  }

  firstABP = false;
  lastSeqABP = pkt->seq;

  Serial.print("START|TYPE:ABP|ABP:");
  Serial.print(pkt->abp_raw);

  Serial.print("|SBP:");
  Serial.print(pkt->sbp_x10 / 10.0, 1);

  Serial.print("|DBP:");
  Serial.print(pkt->dbp_x10 / 10.0, 1);

  Serial.print("|SEQ:");
  Serial.print(pkt->seq);

  Serial.println("|END");

  if (pkt->seq % LOG_INTERVAL == 0) {
    float loss = (totalABP + lostABP) > 0 ?
      (float)lostABP / (totalABP + lostABP) * 100.0f : 0;

    Serial.printf("[RX ABP] SEQ=%u LOSS=%.2f%%\n", pkt->seq, loss);
  }
}


void handlePPG() {
  PPGPacket* pkt = (PPGPacket*) buffer;
  totalPPG++;

  if (!firstPPG) {
    uint16_t expected = lastSeqPPG + 1;
    if (pkt->seq != expected)
      lostPPG += (uint16_t)(pkt->seq - expected);
  }

  firstPPG = false;
  lastSeqPPG = pkt->seq;

  Serial.print("START|TYPE:PPG|PPG:");
  Serial.print(pkt->ppg_raw);

  Serial.print("|SEQ:");
  Serial.print(pkt->seq);

  Serial.println("|END");

  if (pkt->seq % LOG_INTERVAL == 0) {
    float loss = (totalPPG + lostPPG) > 0 ?
      (float)lostPPG / (totalPPG + lostPPG) * 100.0f : 0;

    Serial.printf("[RX PPG] SEQ=%u LOSS=%.2f%%\n", pkt->seq, loss);
  }
}
