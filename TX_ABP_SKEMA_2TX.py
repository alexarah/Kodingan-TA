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

# ─── KONFIGURASI (WAJIB SAMA dengan tx_ppg.py & rx_abp_ppg.py) ─────
FS               = 125
ABP_FILE         = 'p000188_abp.npy'
SEGMENTS = [
    {'idx': 0,  'label': 'Normal'},
    {'idx': 20, 'label': 'BP Tinggi'},
    {'idx': 25, 'label': 'BP Rendah'},
]
PAKET_PER_SEGMEN = 3750          # HARUS SAMA di tx_ppg.py & rx_abp_ppg.py
STARTUP_GRACE_S  = 8.0           # jeda awal agar tx_abp.py & tx_ppg.py bisa mulai berdekatan
# ─────────────────────────────────────────────────────────────

# ─── KONFIGURASI ────────────────────────────────
PORT           = 'COM7'         
BAUD_RATE      = 115200
OUTPUT_DIR     = 'hasil_tx_abp'
WINDOW_S       = 10
PEAK_DIST      = 40
PEAK_PROM      = 3
THROUGHPUT_WIN_S    = 3
TX_PRINT_INTERVAL_S = 1.0
TX_CSV_FLUSH_EVERY  = 25
# ─────────────────────────────────────────────────────────────

SLOT_DURATION_S = PAKET_PER_SEGMEN / FS   # nominal, mis. 3750/125 = 30.0 s
N_SEG = len(SEGMENTS)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ABP_PATH    = os.path.join(SCRIPT_DIR, ABP_FILE)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_PATH, exist_ok=True)

WINDOW_N = WINDOW_S * FS

buf_tx   = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
lock     = threading.Lock()
stop_evt = threading.Event()

stats = {
    'tx_pkt': 0, 'sbp_tx': 0.0, 'dbp_tx': 0.0,
    'bytes_tx': 0, 'thr_tx_bps': 0.0, 'thr_tx_kbps': 0.0, 'pkt_rate_tx': 0.0,
    'segmen_label': '-', 'menunggu_slot': True, 'selesai': False,
}


class ThroughputMeter:
    def __init__(self, window_s=THROUGHPUT_WIN_S):
        self.window_s = window_s
        self._samples = deque()
        self._lock    = threading.Lock()

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


meter_tx = ThroughputMeter()


def load_segment_row(filepath, seg_idx):
    data = np.load(filepath, allow_pickle=True)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_total = data.shape[0]
    if not (0 <= seg_idx < n_total):
        raise ValueError(f"Segmen idx {seg_idx} di luar jangkauan (0..{n_total-1})")
    return data[seg_idx].astype(float)


def hitung_sbp_dbp(buf_list):
    if len(buf_list) < PEAK_DIST * 2:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(buf_list, dtype=float)
    peaks,   _ = find_peaks( arr, distance=PEAK_DIST, prominence=PEAK_PROM)
    valleys, _ = find_peaks(-arr, distance=PEAK_DIST, prominence=PEAK_PROM)
    sbp = float(arr[peaks].mean())   if len(peaks)   > 0 else 0.0
    dbp = float(arr[valleys].mean()) if len(valleys) > 0 else 0.0
    pp  = sbp - dbp if sbp > 0 and dbp > 0 else 0.0
    map_ = dbp + pp / 3.0 if pp > 0 else 0.0
    return sbp, dbp, map_, pp


def slot_mulai_ke(seg_i):
    # Urutan slot: ABP0, PPG0, ABP1, PPG1, ... → slot ABP segmen ke-i ada di indeks 2*i
    return SLOT_DURATION_S * (2 * seg_i)


def simpan_ringkasan(seq_terkirim, n_target, elapsed_total_s, waktu_aktif_total_s, snapshot):
    # Metrik berbasis DURASI SESI (termasuk waktu nunggu giliran slot PPG)
    avg_kbps     = (snapshot['bytes_tx'] * 8 / elapsed_total_s / 1000.0) if elapsed_total_s > 0 else 0.0
    avg_pkt_rate = (seq_terkirim / elapsed_total_s) if elapsed_total_s > 0 else 0.0
    persen_kirim = (seq_terkirim / n_target * 100) if n_target > 0 else 0.0

    # Metrik berbasis WAKTU AKTIF KIRIM SAJA (tanpa waktu nunggu slot PPG/jeda awal)
    # -> ini yang mencerminkan kecepatan pengiriman sebenarnya (mendekati FS)
    avg_kbps_aktif     = (snapshot['bytes_tx'] * 8 / waktu_aktif_total_s / 1000.0) if waktu_aktif_total_s > 0 else 0.0
    avg_pkt_rate_aktif = (seq_terkirim / waktu_aktif_total_s) if waktu_aktif_total_s > 0 else 0.0
    persen_waktu_aktif = (waktu_aktif_total_s / elapsed_total_s * 100) if elapsed_total_s > 0 else 0.0

    ring_path = os.path.join(OUTPUT_PATH, 'ringkasan_tx_abp.csv')
    with open(ring_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metrik', 'nilai'])
        w.writerow(['port', PORT])
        w.writerow(['sumber_file', ABP_FILE])
        w.writerow(['segmen', [s['label'] for s in SEGMENTS]])
        w.writerow(['paket_per_segmen', PAKET_PER_SEGMEN])
        w.writerow(['fs_hz', FS])
        w.writerow(['target_paket', n_target])
        w.writerow(['paket_terkirim', seq_terkirim])
        w.writerow(['persen_terkirim', round(persen_kirim, 2)])
        w.writerow(['durasi_sesi_s', round(elapsed_total_s, 3)])
        w.writerow(['bytes_terkirim', snapshot['bytes_tx']])
        w.writerow([])
        w.writerow(['--- metrik rata2 vs DURASI SESI (termasuk nunggu slot PPG) ---', ''])
        w.writerow(['throughput_rata2_kbps', round(avg_kbps, 3)])
        w.writerow(['laju_paket_rata2_pps', round(avg_pkt_rate, 2)])
        w.writerow([])
        w.writerow(['--- metrik rata2 vs WAKTU AKTIF KIRIM SAJA (harusnya ~FS) ---', ''])
        w.writerow(['waktu_aktif_kirim_s', round(waktu_aktif_total_s, 3)])
        w.writerow(['persen_waktu_aktif_dari_sesi', round(persen_waktu_aktif, 2)])
        w.writerow(['throughput_aktif_kbps', round(avg_kbps_aktif, 3)])
        w.writerow(['laju_paket_aktif_pps', round(avg_pkt_rate_aktif, 2)])
        w.writerow([])
        w.writerow(['sbp_terakhir', round(snapshot['sbp_tx'], 1)])
        w.writerow(['dbp_terakhir', round(snapshot['dbp_tx'], 1)])
    print(f"[TX-ABP] Ringkasan -> {ring_path}")


# ─── THREAD TX ───────────────────────────────────────────────
def thread_tx():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
        time.sleep(1.5)
        print(f"[TX-ABP] Terhubung ke {PORT}")
    except Exception as e:
        print(f"[TX-ABP] ERROR: {e}")
        stop_evt.set()
        return

    csv_path = os.path.join(OUTPUT_PATH, 'tx_abp.csv')
    f = open(csv_path, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['timestamp_s', 'segmen_idx', 'segmen_label', 'seq',
                'abp_raw', 'abp_mmhg', 'sbp_tx', 'dbp_tx',
                'bytes_tx', 'bytes_tx_total',
                'thr_tx_bps', 'thr_tx_kbps', 'pkt_rate_tx'])

    interval = 1.0 / FS
    seq      = 0
    last_print_t = time.perf_counter()
    waktu_aktif_total_s = 0.0   # akumulasi waktu yang benar-benar dipakai mengirim (di luar waktu nunggu slot)

    t0 = time.perf_counter()
    print(f"[TX-ABP] Menunggu {STARTUP_GRACE_S:.0f}s jeda awal — "
          f"jalankan tx_ppg.py sekarang kalau belum!")
    n_target_total = PAKET_PER_SEGMEN * N_SEG

    for i, seg in enumerate(SEGMENTS):
        if stop_evt.is_set():
            break

        target_start = t0 + STARTUP_GRACE_S + slot_mulai_ke(i)
        with lock:
            stats['menunggu_slot'] = True
            stats['segmen_label']  = f"menunggu slot ABP ({seg['label']})"

        wait_s = target_start - time.perf_counter()
        if wait_s > 0:
            print(f"[TX-ABP] Menunggu slot ABP segmen '{seg['label']}' ({wait_s:.1f}s lagi)...")
            time.sleep(wait_s)
        else:
            print(f"[TX-ABP] PERINGATAN: telat {-wait_s:.1f}s masuk slot '{seg['label']}', mulai sekarang.")

        with lock:
            stats['menunggu_slot'] = False
            stats['segmen_label']  = f"MENGIRIM: ABP ({seg['label']})"

        sig = load_segment_row(ABP_PATH, seg['idx'])
        n   = len(sig)
        buf_win = []
        sbp_val = dbp_val = 0.0
        next_t = time.perf_counter()
        segmen_mulai_t = time.perf_counter()   # tanda waktu mulai AKTIF kirim segmen ini

        for k in range(PAKET_PER_SEGMEN):
            if stop_evt.is_set():
                break

            val = float(sig[k % n])
            raw = int(round(val * 100))

            buf_win.append(val)
            if len(buf_win) > FS * 5: buf_win.pop(0)
            if k % FS == 0:
                sbp_val, dbp_val, _, _ = hitung_sbp_dbp(buf_win)

            packet    = f"START|TYPE:ABP|ABP:{raw}|SBP:{sbp_val:.1f}|DBP:{dbp_val:.1f}|SEQ:{seq}|END\n"
            pkt_bytes = len(packet.encode('utf-8'))

            try:
                ser.write(packet.encode('utf-8'))
            except Exception as e:
                print(f"[TX-ABP] Write error: {e}"); break

            meter_tx.update(pkt_bytes)
            bps_tx, pps_tx = meter_tx.get()

            with lock:
                buf_tx.append(val)
                stats['tx_pkt']      += 1
                stats['sbp_tx']      = sbp_val
                stats['dbp_tx']      = dbp_val
                stats['bytes_tx']   += pkt_bytes
                stats['thr_tx_bps']  = bps_tx
                stats['thr_tx_kbps'] = bps_tx / 1000.0
                stats['pkt_rate_tx'] = pps_tx
                bytes_tx_total_snap = stats['bytes_tx']

            w.writerow([round(time.perf_counter() - t0, 3), seg['idx'], seg['label'], seq,
                        raw, round(val, 2), round(sbp_val, 1), round(dbp_val, 1),
                        pkt_bytes, bytes_tx_total_snap,
                        round(bps_tx, 1), round(bps_tx / 1000.0, 3), round(pps_tx, 2)])

            now_t = time.perf_counter()
            if now_t - last_print_t >= TX_PRINT_INTERVAL_S:
                print(f"[TX-ABP] seg={seg['label']:<10} {k+1:>5}/{PAKET_PER_SEGMEN} | "
                      f"{pps_tx:.1f} pkt/s | {bps_tx/1000:.2f} kbps")
                last_print_t = now_t

            seq    += 1
            next_t += interval
            sleep   = next_t - time.perf_counter()
            if sleep > 0: time.sleep(sleep)
            else: next_t = time.perf_counter()

        waktu_aktif_total_s += (time.perf_counter() - segmen_mulai_t)   # akumulasi durasi AKTIF segmen ini
        f.flush()
        print(f"[TX-ABP] Segmen {seg['label']} selesai — {PAKET_PER_SEGMEN} paket.")

    f.close()
    ser.close()

    elapsed_total_s = time.perf_counter() - t0
    with lock:
        snapshot = dict(stats)
        stats['selesai'] = True
        stats['segmen_label'] = 'SELESAI'

    print(f"[TX-ABP] Selesai — total {seq:,} paket. CSV -> {csv_path}")
    print(f"[TX-ABP] Waktu aktif kirim: {waktu_aktif_total_s:.1f}s dari total sesi {elapsed_total_s:.1f}s "
          f"(rate aktif ~{(seq / waktu_aktif_total_s if waktu_aktif_total_s > 0 else 0):.1f} pkt/s)")
    simpan_ringkasan(seq, n_target_total, elapsed_total_s, waktu_aktif_total_s, snapshot)

    time.sleep(2.0)
    stop_evt.set()


# ─── DASHBOARD ───────────────────────────────────────────────
def run_dashboard():
    plt.rcParams.update({
        'figure.facecolor': '#0D1117', 'axes.facecolor': '#161B22',
        'axes.edgecolor': '#30363D',   'axes.labelcolor': '#C9D1D9',
        'xtick.color': '#8B949E',      'ytick.color': '#8B949E',
        'grid.color': '#21262D',       'text.color': '#C9D1D9',
        'font.family': 'monospace',
    })

    t_axis = np.linspace(0, WINDOW_S, WINDOW_N)
    fig = plt.figure(figsize=(11, 8))
    sup_title = fig.suptitle(f'ABP Monitor — TX SAJA  |  Port: {PORT}',
                              color='#58A6FF', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                           left=0.09, right=0.96, top=0.87, bottom=0.08,
                           height_ratios=[1.4, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    ax1.set_title('Sinyal ABP - TX (yang dikirim)', color='#58A6FF', fontsize=10)
    ax1.set_xlim(0, WINDOW_S)
    ax1.set_ylim(30, 175)
    ax1.set_xlabel('Waktu (detik)', fontsize=9)
    ax1.set_ylabel('mmHg', fontsize=9)
    ax1.grid(True, alpha=0.3)

    line_tx, = ax1.plot(t_axis, list(buf_tx), color='#58A6FF', lw=1.0)

    stats_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                          fontsize=10.5, verticalalignment='top',
                          fontfamily='monospace', color='#E3B341',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22', alpha=0.85))

    def update(_frame):
        if stop_evt.is_set():
            try: ani.event_source.stop()
            except Exception: pass
            plt.close(fig)
            return

        with lock:
            tx = np.array(buf_tx)
            s = dict(stats)

        line_tx.set_ydata(tx)
        if np.ptp(tx) > 0:
            margin = np.ptp(tx) * 0.12
            ax1.set_ylim(np.min(tx) - margin, np.max(tx) + margin)

        n_target_total = PAKET_PER_SEGMEN * N_SEG
        progress_pct = (s['tx_pkt'] / n_target_total * 100) if n_target_total > 0 else 0
        txt = (
            f"Status     : {s['segmen_label']}\n"
            f"TX paket   : {s['tx_pkt']:>8,} / {n_target_total:,}  ({progress_pct:5.1f} %)\n"
            f"Throughput : {s['thr_tx_kbps']:>7.2f} kbps\n"
            f"Laju paket : {s['pkt_rate_tx']:>7.1f} pkt/s\n"
            f"SBP / DBP  : {s['sbp_tx']:>5.1f} / {s['dbp_tx']:<5.1f} mmHg\n"
            f"Byte TX    : {s['bytes_tx']/1024:>7.1f} KB"
            + ('\n\n SELESAI' if s['selesai'] else '')
        )
        stats_text.set_text(txt)
        fig.canvas.draw_idle()

    ani = animation.FuncAnimation(fig, update, interval=150, blit=False, cache_frame_data=False)

    def on_close(event):
        try: ani.event_source.stop()
        except Exception: pass
        stop_evt.set()

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()


def main():
    n_target_total = PAKET_PER_SEGMEN * N_SEG
    print('=' * 70)
    print('  TX ABP SAJA — TDM berjadwal (proses terpisah dari TX PPG & RX)')
    print('=' * 70)
    print(f'  Port          : {PORT}')
    print(f'  Segmen        : {[s["label"] for s in SEGMENTS]}')
    print(f'  Paket/segmen  : {PAKET_PER_SEGMEN:,}  (durasi slot ~{SLOT_DURATION_S:.1f}s)')
    print(f'  Target total  : {n_target_total:,} paket')
    print(f'  Jeda awal     : {STARTUP_GRACE_S:.0f}s')
    print(f'  Output        : {OUTPUT_PATH}/')
    print('=' * 70)
    print('\nPENTING: jalankan tx_ppg.py juga, sedekat mungkin waktunya dengan script ini!')
    print('Tutup jendela dashboard untuk berhenti paksa.\n')

    if not os.path.exists(ABP_PATH):
        print(f" ERROR: File ABP tidak ditemukan!\n   Path: {ABP_PATH}")
        return

    t = threading.Thread(target=thread_tx, daemon=True)
    t.start()

    try:
        run_dashboard()
    except KeyboardInterrupt:
        print('\n[Main] Dihentikan.')
        stop_evt.set()

    t.join(timeout=5)
    print('[Main] Selesai.')


if __name__ == '__main__':
    main()