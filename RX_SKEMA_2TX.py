"""
tdm_rx.py — TDM RX-ONLY (ABP + PPG) + HR + SNR + BP + LOSS/SUCCESS RATIO
Bagian RX dari tdm_4panel_metrics_final_oneshot.py (dipisah dari TX).

CATATAN PENTING (akibat pisah jadi 2 proses/laptop):
- SNR ABP & PPG dihitung dari SINYAL REFERENSI LOKAL (dimuat ulang dari
  p000188_abp.npy & p000188_ppg.npy di laptop RX ini), dicocokkan lewat
  nomor SEQ — bukan lagi baca buffer TX langsung. Syaratnya: file .npy,
  SEGMENTS, SEGMENT_DURATION, FS di sini HARUS identik dengan tdm_tx.py.
- Segmen & mode aktif (ABP/PPG) dihitung SENDIRI oleh RX dari nomor SEQ,
  bukan dari field khusus di paket (karena protokolnya memang tidak
  membawa info itu).
- Loss & Success Ratio dihitung berbasis TOTAL_TARGET (angka pasti dari
  konfigurasi), bukan dari stats['tx_pkt'] proses TX yang sudah terpisah.
- SBP/DBP/MAP sekarang dihitung dari sinyal yang BENAR-BENAR diterima RX
  (buf_rx_abp), bukan dari buffer sisi TX seperti versi gabungan asli —
  jadi sekarang ikut mencerminkan efek packet loss & noise transmisi.
"""

import serial
import threading
import time
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.signal import find_peaks
from collections import deque
from datetime import datetime

# ─── KONFIGURASI (harus SAMA dengan tdm_tx.py, kecuali PORT) ──
PORT_RX = 'COM14'
BAUD_RATE = 115200
FS = 125
WINDOW_S = 5
SEGMENT_DURATION = 15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABP_NPY_PATH = os.path.join(SCRIPT_DIR, 'p000188_abp.npy')
PPG_NPY_PATH = os.path.join(SCRIPT_DIR, 'p000188_ppg.npy')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'hasil_TDM')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEGMENTS = [
    {'idx': 0, 'label': 'Normal'},
    {'idx': 20, 'label': 'BP Tinggi'},
    {'idx': 25, 'label': 'BP Rendah'},
]
# ──────────────────────────────────────────────────────────────

HR_MIN = 40
HR_MAX = 160
WINDOW_N = WINDOW_S * FS
HR_WINDOW = 60

THROUGHPUT_WIN_S = 3
IDLE_TIMEOUT_S = 6.0     # auto-berhenti kalau tidak ada paket masuk selama ini
SNR_WINDOW_N = 625       # jumlah sampel terakhir (per channel) buat SNR bergerak

TARGET_N = int(SEGMENT_DURATION * FS)          # 1.875 paket / mode / segmen
TOTAL_TARGET = TARGET_N * 2 * len(SEGMENTS)     # total paket keseluruhan (11.250)

# ─── SETUP FILE CSV (RX ABP & RX PPG) ─────────────────────────
TIMESTAMP_RUN = datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_RX_ABP_FILENAME = os.path.join(OUTPUT_DIR, f"rx_abp_{TIMESTAMP_RUN}.csv")
CSV_RX_PPG_FILENAME = os.path.join(OUTPUT_DIR, f"rx_ppg_{TIMESTAMP_RUN}.csv")
RINGKASAN_FILENAME  = os.path.join(OUTPUT_DIR, f"ringkasan_RX_{TIMESTAMP_RUN}.csv")

CSV_HEADER = ['timestamp', 'seq', 'segmen_idx', 'segmen_label', 'raw', 'mmhg']

csv_file_rx_abp = open(CSV_RX_ABP_FILENAME, mode='w', newline='', encoding='utf-8')
csv_writer_rx_abp = csv.writer(csv_file_rx_abp)
csv_writer_rx_abp.writerow(CSV_HEADER)

csv_file_rx_ppg = open(CSV_RX_PPG_FILENAME, mode='w', newline='', encoding='utf-8')
csv_writer_rx_ppg = csv.writer(csv_file_rx_ppg)
csv_writer_rx_ppg.writerow(CSV_HEADER)

csv_lock = threading.Lock()
# ─────────────────────────────────────────────────────────────

buf_rx_abp = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
buf_rx_ppg = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)

hr_abp_hist = deque([0.0] * HR_WINDOW, maxlen=HR_WINDOW)
hr_ppg_hist = deque([0.0] * HR_WINDOW, maxlen=HR_WINDOW)

snr_expected_abp = deque(maxlen=SNR_WINDOW_N)
snr_actual_abp   = deque(maxlen=SNR_WINDOW_N)
snr_expected_ppg = deque(maxlen=SNR_WINDOW_N)
snr_actual_ppg   = deque(maxlen=SNR_WINDOW_N)

lock = threading.Lock()
stop_evt = threading.Event()
start_time = time.time()

stats = {
    'mode': 'ABP',
    'segmen': 0,
    'segmen_label': 'Normal',
    'rx_pkt': 0,
    'loss_pkt': 0,
    'loss_pct': 0.0,
    'success_ratio': 0.0,
    'hr_abp': 0.0,
    'hr_ppg': 0.0,
    'sbp': 0.0,
    'dbp': 0.0,
    'map': 0.0,
    'snr_abp_db': 0.0,
    'snr_ppg_db': 0.0,
    'bytes_rx': 0,
    'thr_kbps': 0.0,
    'pkt_rate': 0.0,
    'selesai': False,
}


class ThroughputMeter:
    def __init__(self, window_s=THROUGHPUT_WIN_S):
        self.window_s = window_s
        self._samples = deque()
        self._lock = threading.Lock()

    def update(self, n_bytes, n_pkts=1):
        now = time.perf_counter()
        with self._lock:
            self._samples.append((now, n_bytes, n_pkts))
            cutoff = now - self.window_s
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def get(self):
        now = time.perf_counter()
        with self._lock:
            if len(self._samples) < 2:
                return 0.0, 0.0
            elapsed = now - self._samples[0][0]
            if elapsed <= 0:
                return 0.0, 0.0
            total_b = sum(s[1] for s in self._samples)
            total_p = sum(s[2] for s in self._samples)
            return (total_b * 8) / elapsed, total_p / elapsed


meter = ThroughputMeter()


# ─── FUNGSI ──────────────────────────────────────────────────
def load_abp(seg_idx):
    data = np.load(ABP_NPY_PATH)
    seg = data[seg_idx].astype(float)
    return np.tile(seg, 100)

def load_ppg(seg_idx):
    data = np.load(PPG_NPY_PATH, allow_pickle=True)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    seg = data[seg_idx].astype(float) if data.shape[0] > seg_idx else data.flatten().astype(float)
    return np.tile(seg, 100)

def bangun_referensi():
    """Bangun array referensi (panjang TOTAL_TARGET) yang berisi nilai
    IDEAL untuk tiap nomor SEQ — persis mengikuti urutan pengiriman TX:
    segmen0-ABP(1875) -> segmen0-PPG(1875) -> segmen1-ABP(1875) -> ...
    Dipakai untuk hitung SNR & menentukan segmen/mode aktif TANPA perlu
    akses ke proses TX."""
    ref = np.zeros(TOTAL_TARGET, dtype=float)
    for i, seg in enumerate(SEGMENTS):
        base = i * 2 * TARGET_N
        abp_sig = load_abp(seg['idx'])
        n_abp = len(abp_sig)
        for j in range(TARGET_N):
            ref[base + j] = abp_sig[j % n_abp]

        ppg_sig = load_ppg(seg['idx'])
        n_ppg = len(ppg_sig)
        for j in range(TARGET_N):
            ref[base + TARGET_N + j] = ppg_sig[j % n_ppg]
    return ref

def info_dari_seq(seq):
    """Tentukan (segmen_idx, segmen_label, mode) HANYA dari nomor SEQ +
    konfigurasi lokal RX. Blok ke-0..5: seg0-ABP, seg0-PPG, seg1-ABP,
    seg1-PPG, seg2-ABP, seg2-PPG (mengikuti urutan pengiriman TX)."""
    block = min(seq // TARGET_N, 2 * len(SEGMENTS) - 1)
    seg_pos = block // 2
    mode = 'ABP' if block % 2 == 0 else 'PPG'
    seg = SEGMENTS[seg_pos]
    return seg['idx'], seg['label'], mode

def hitung_hr(buf_list, fs=FS, height=0.35, distance_s=0.35, prominence=0.2):
    arr = np.array(buf_list, dtype=float)
    if len(arr) < fs:
        return 0.0
    rng = arr.max() - arr.min()
    if rng < 0.05:
        return 0.0
    arr = np.convolve(arr, np.ones(5)/5, mode='same')
    arr_n = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    peaks, _ = find_peaks(arr_n, height=height,
                           distance=int(distance_s * fs),
                           prominence=prominence)
    if len(peaks) < 2:
        return 0.0
    rr = np.diff(peaks) / fs
    hr = 60.0 / rr
    hr = hr[(hr >= HR_MIN) & (hr <= HR_MAX)]
    return float(np.median(hr)) if len(hr) > 0 else 0.0

def hitung_bp(buf_list):
    arr = np.array(buf_list, dtype=float)
    if len(arr) < 2:
        return 0.0, 0.0, 0.0
    sbp = float(np.percentile(arr, 95))
    dbp = float(np.percentile(arr, 5))
    return sbp, dbp, float(dbp + (sbp - dbp) / 3.0)

def hitung_snr(expected_hist, actual_hist):
    n = min(len(expected_hist), len(actual_hist))
    if n < 10:
        return 0.0
    exp_arr = np.array(expected_hist)[-n:]
    act_arr = np.array(actual_hist)[-n:]
    ps = np.mean(exp_arr ** 2)
    pn = np.mean((exp_arr - act_arr) ** 2)
    if pn <= 1e-12:
        return 99.0
    return float(10 * np.log10(ps / pn))

def log_csv(channel, seq, segmen_idx, segmen_label, raw, mmhg=''):
    timestamp = time.time() - start_time
    row = [
        f"{timestamp:.3f}",
        seq,
        segmen_idx,
        segmen_label,
        int(raw),
        f"{mmhg:.2f}" if mmhg != '' else ''
    ]
    with csv_lock:
        if channel == 'ABP':
            csv_writer_rx_abp.writerow(row)
        else:
            csv_writer_rx_ppg.writerow(row)

# ─── RX THREAD ───────────────────────────────────────────────
def thread_rx():
    try:
        print("[RX] Memuat sinyal referensi (untuk SNR & deteksi segmen) ...")
        ref = bangun_referensi()
        print(f"[RX] Referensi siap: {len(ref):,} sampel (TOTAL_TARGET = {TOTAL_TARGET:,})")
    except Exception as e:
        print(f"[RX] ⚠️  Gagal memuat referensi ({e}) — SNR tidak akan dihitung.")
        ref = None

    try:
        ser = serial.Serial(PORT_RX, BAUD_RATE, timeout=1)
        time.sleep(2.0)
        ser.reset_input_buffer()
        print(f"[RX] OK - {PORT_RX}")
    except Exception as e:
        print(f"[RX] ERROR: {e}")
        return

    serial_buf = b""
    last_rx_time = time.time()
    started_receiving = False

    while not stop_evt.is_set():
        if started_receiving and (time.time() - last_rx_time > IDLE_TIMEOUT_S):
            print(f"\n[RX] Tidak ada data baru selama {IDLE_TIMEOUT_S:.0f} detik — anggap TX sudah selesai.")
            with lock:
                stats['selesai'] = True
            break

        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            time.sleep(0.001)
            continue

        serial_buf += chunk

        while b"\n" in serial_buf:
            line, serial_buf = serial_buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue

            try:
                text = line.decode('utf-8', errors='replace').strip()
                if not text.startswith('START'):
                    continue

                parts = {}
                for tok in text.split('|'):
                    if ':' in tok:
                        k, v = tok.split(':', 1)
                        parts[k.strip()] = v.strip()

                if 'SLOT' not in parts:
                    continue

                slot = int(parts['SLOT'])
                seq = int(parts.get('SEQ', 0))

                last_rx_time = time.time()
                started_receiving = True

                seg_idx, seg_label, mode_now = info_dari_seq(seq)

                n_bytes_line = len(line) + 1
                meter.update(n_bytes_line, n_pkts=1)
                bps, pps = meter.get()

                raw_val = None

                with lock:
                    stats['rx_pkt'] += 1
                    stats['bytes_rx'] += n_bytes_line
                    stats['thr_kbps'] = bps / 1000.0
                    stats['pkt_rate'] = pps
                    stats['segmen'] = seg_idx
                    stats['segmen_label'] = seg_label
                    stats['mode'] = mode_now

                    stats['loss_pkt'] = max(0, TOTAL_TARGET - stats['rx_pkt'])
                    stats['loss_pct'] = (stats['loss_pkt'] / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0.0
                    stats['success_ratio'] = (stats['rx_pkt'] / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0.0

                    if slot == 0 and 'ABP' in parts:
                        raw_val = int(parts['ABP'])
                        val = raw_val / 100.0
                        buf_rx_abp.append(val)
                        if ref is not None and 0 <= seq < len(ref):
                            snr_expected_abp.append(float(ref[seq]))
                            snr_actual_abp.append(val)
                            stats['snr_abp_db'] = hitung_snr(snr_expected_abp, snr_actual_abp)
                    elif slot == 1 and 'PPG' in parts:
                        raw_val = int(parts['PPG'])
                        val = raw_val / 100.0
                        buf_rx_ppg.append(val)
                        if ref is not None and 0 <= seq < len(ref):
                            snr_expected_ppg.append(float(ref[seq]))
                            snr_actual_ppg.append(val)
                            stats['snr_ppg_db'] = hitung_snr(snr_expected_ppg, snr_actual_ppg)
                    else:
                        val = None

                    if stats['rx_pkt'] % FS == 0:
                        if len(buf_rx_abp) >= FS:
                            hr = hitung_hr(list(buf_rx_abp)[-FS:])
                            stats['hr_abp'] = hr
                            if HR_MIN <= hr <= HR_MAX:
                                hr_abp_hist.append(hr)
                            else:
                                hr_abp_hist.append(hr_abp_hist[-1] if len(hr_abp_hist) > 0 else 0.0)

                            # FIX: SBP/DBP/MAP sekarang dari sinyal yang
                            # BENAR-BENAR diterima RX (buf_rx_abp), bukan
                            # dari buffer sisi TX seperti versi gabungan asli.
                            sbp, dbp, mp = hitung_bp(list(buf_rx_abp)[-FS:])
                            stats['sbp'] = sbp
                            stats['dbp'] = dbp
                            stats['map'] = mp

                        if len(buf_rx_ppg) >= FS:
                            hr = hitung_hr(list(buf_rx_ppg)[-FS:], height=0.1, distance_s=0.4, prominence=0.1)
                            stats['hr_ppg'] = hr
                            if HR_MIN <= hr <= HR_MAX:
                                hr_ppg_hist.append(hr)
                            else:
                                hr_ppg_hist.append(hr_ppg_hist[-1] if len(hr_ppg_hist) > 0 else 0.0)

                if raw_val is not None:
                    channel = 'ABP' if slot == 0 else 'PPG'
                    mmhg_rx = val if slot == 0 else ''
                    log_csv(channel, seq, seg_idx, seg_label, raw_val, mmhg_rx)

            except Exception:
                continue

    ser.close()

# ─── DASHBOARD ──────────────────────────────────────────────
def run_dashboard():
    plt.rcParams.update({
        'figure.facecolor': '#0D1117',
        'axes.facecolor': '#161B22',
        'axes.edgecolor': '#30363D',
        'axes.labelcolor': '#C9D1D9',
        'xtick.color': '#8B949E',
        'ytick.color': '#8B949E',
        'grid.color': '#21262D',
        'text.color': '#C9D1D9',
    })

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                           left=0.08, right=0.97, top=0.90, bottom=0.06,
                           height_ratios=[1, 1, 1.3])

    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('📥 RX ABP', color='#FF6B6B', fontsize=11, fontweight='bold', loc='left')
    ax1.set_xlim(0, WINDOW_S)
    ax1.set_ylim(-1.5, 1.5)
    ax1.grid(True, alpha=0.2)
    ax1.axhline(y=0, color='gray', lw=0.5, linestyle='--', alpha=0.5)
    line_rx_abp, = ax1.plot([], [], 'r-', lw=2.0)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('📥 RX PPG', color='#51CF66', fontsize=11, fontweight='bold', loc='left')
    ax2.set_xlim(0, WINDOW_S)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.2)
    ax2.axhline(y=0, color='gray', lw=0.5, linestyle='--', alpha=0.5)
    line_rx_ppg, = ax2.plot([], [], 'g-', lw=2.0)

    ax_stats = fig.add_subplot(gs[2])
    ax_stats.axis('off')

    t_axis = np.linspace(0, WINDOW_S, WINDOW_N)

    mode_text = fig.text(0.5, 0.955, '📡 MENUNGGU DATA RX...', color='#FFD700', fontsize=14,
                          fontweight='bold', ha='center', transform=fig.transFigure)

    stats_text = ax_stats.text(0.02, 0.95, '', transform=ax_stats.transAxes,
                                fontsize=10.5, fontfamily='monospace', color='#E3B341',
                                verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22', alpha=0.85))

    def on_close(event):
        stop_evt.set()

    fig.canvas.mpl_connect('close_event', on_close)

    def update(frame):
        with lock:
            rx_abp = np.array(buf_rx_abp, dtype=float)
            rx_ppg = np.array(buf_rx_ppg, dtype=float)
            s = dict(stats)

        for data, line, ax in [(rx_abp, line_rx_abp, ax1), (rx_ppg, line_rx_ppg, ax2)]:
            if len(data) == WINDOW_N:
                line.set_data(t_axis, data)
                rng = data.max() - data.min()
                if rng > 0.05:
                    margin = rng * 0.15
                    ax.set_ylim(data.min() - margin, data.max() + margin)

        if s.get('selesai'):
            mode_text.set_text(f'✅ SELESAI — Segmen terakhir: {s["segmen"]} ({s["segmen_label"]})')
            mode_text.set_color('#58A6FF')
        else:
            mode_text.set_text(f'{"🔴" if s["mode"] == "ABP" else "🟢"} MODE: {s["mode"]} — Segmen {s["segmen"]} ({s["segmen_label"]})')
            mode_text.set_color('#FF6B6B' if s['mode'] == 'ABP' else '#51CF66')

        C = "   |   "
        stats_text.set_text(
            f"HR ABP    : {s['hr_abp']:>6.1f} BPM{C}HR PPG    : {s['hr_ppg']:>6.1f} BPM\n"
            f"SBP       : {s['sbp']:>6.1f} mmHg{C}DBP       : {s['dbp']:>6.1f} mmHg{C}MAP : {s['map']:>6.1f} mmHg\n"
            f"SNR ABP   : {s['snr_abp_db']:>6.2f} dB{C}SNR PPG   : {s['snr_ppg_db']:>6.2f} dB\n"
            f"Loss      : {s['loss_pct']:>6.2f} %{C}Success Ratio: {s['success_ratio']:>6.1f} %\n"
            f"Throughput: {s['thr_kbps']:>6.3f} kbps{C}Laju paket: {s['pkt_rate']:>6.1f} pkt/s\n"
            f"RX Paket  : {s['rx_pkt']:>8,} / {TOTAL_TARGET:,}{C}Loss pkt: {s['loss_pkt']:,}\n"
            f"(Tutup jendela ini untuk berhenti & simpan CSV)"
        )

        if stop_evt.is_set():
            ani.event_source.stop()
            plt.close(fig)

        return (line_rx_abp, line_rx_ppg, mode_text, stats_text)

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
    plt.show()
    stop_evt.set()

# ─── MAIN ────────────────────────────────────────────────────
def main():
    print('=' * 60)
    print('  TDM RX-ONLY (ABP + PPG) + HR + SNR + BP + LOSS/SUCCESS')
    print(f'  Target total paket: {TOTAL_TARGET:,}')
    print('  Tutup jendela dashboard untuk berhenti manual')
    print(f'  Data RX ABP akan disimpan ke: {CSV_RX_ABP_FILENAME}')
    print(f'  Data RX PPG akan disimpan ke: {CSV_RX_PPG_FILENAME}')
    print('=' * 60)

    t = threading.Thread(target=thread_rx, daemon=True)
    t.start()

    time.sleep(1)
    run_dashboard()

    time.sleep(0.5)
    csv_file_rx_abp.close()
    csv_file_rx_ppg.close()

    print('\n' + '=' * 60)
    print('  📊 RINGKASAN HASIL PENERIMAAN (RX)')
    print('=' * 60)
    print(f"  RX Paket total   : {stats['rx_pkt']:,} / {TOTAL_TARGET:,}")
    print(f"  Paket hilang     : {stats['loss_pkt']:,}")
    print(f"  Packet Loss Rate : {stats['loss_pct']:.2f} %")
    print(f"  Success Ratio    : {stats['success_ratio']:.2f} %")
    print(f"  Throughput       : {stats['thr_kbps']:.3f} kbps")
    print(f"  Laju paket       : {stats['pkt_rate']:.1f} pkt/s")
    print(f"  SNR ABP          : {stats['snr_abp_db']:.2f} dB")
    print(f"  SNR PPG          : {stats['snr_ppg_db']:.2f} dB")
    print(f"  HR ABP (terakhir): {stats['hr_abp']:.1f} BPM")
    print(f"  HR PPG (terakhir): {stats['hr_ppg']:.1f} BPM")
    print(f"  SBP/DBP/MAP      : {stats['sbp']:.1f} / {stats['dbp']:.1f} / {stats['map']:.1f} mmHg")
    print('-' * 60)
    print(f"  💾 Data RX ABP tersimpan di: {os.path.abspath(CSV_RX_ABP_FILENAME)}")
    print(f"  💾 Data RX PPG tersimpan di: {os.path.abspath(CSV_RX_PPG_FILENAME)}")
    print('=' * 60)

    with open(RINGKASAN_FILENAME, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metrik', 'nilai'])
        w.writerow(['total_target_paket', TOTAL_TARGET])
        w.writerow(['rx_pkt_total', stats['rx_pkt']])
        w.writerow(['loss_pkt', stats['loss_pkt']])
        w.writerow(['loss_pct', round(stats['loss_pct'], 2)])
        w.writerow(['success_ratio_pct', round(stats['success_ratio'], 2)])
        w.writerow(['throughput_kbps', round(stats['thr_kbps'], 3)])
        w.writerow(['pkt_rate', round(stats['pkt_rate'], 1)])
        w.writerow(['snr_abp_db', round(stats['snr_abp_db'], 2)])
        w.writerow(['snr_ppg_db', round(stats['snr_ppg_db'], 2)])
        w.writerow(['bytes_rx', stats['bytes_rx']])
        w.writerow(['hr_abp_terakhir', round(stats['hr_abp'], 1)])
        w.writerow(['hr_ppg_terakhir', round(stats['hr_ppg'], 1)])
        w.writerow(['sbp_mmhg', round(stats['sbp'], 1)])
        w.writerow(['dbp_mmhg', round(stats['dbp'], 1)])
        w.writerow(['map_mmhg', round(stats['map'], 1)])
    print(f"  💾 Ringkasan RX tersimpan di: {os.path.abspath(RINGKASAN_FILENAME)}")
    print("Program RX selesai.")

if __name__ == '__main__':
    main()