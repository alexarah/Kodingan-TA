"""
tx_ppg_only.py — SISI TX SAJA (untuk setup 2 laptop/proses terpisah)
======================================================================
Jalankan skrip ini di laptop/terminal yang tersambung ke ESP32 TX (USB serial).
Pasangannya: rx_ppg_only.py (dijalankan di laptop/terminal LAIN, tersambung
ke ESP32 RX).

PENTING — HARUS SAMA PERSIS dengan rx_ppg_only.py:
    FS, PPG_FILE, SEGMEN_DIPILIH
Kalau salah satu beda, RX akan salah menyimpulkan target jumlah paket dan
nilai sinyal asli untuk SNR (karena RX memuat file .npy sendiri dan
mencocokkan lewat SEQ, bukan lewat komunikasi langsung ke TX).

Yang beda cuma PORT (port serial ESP32 TX di laptop kamu).

Output:
    hasil_ppg_tx/tx_data.csv      <- detail tiap paket yang dikirim
    hasil_ppg_tx/ringkasan_tx.csv <- ringkasan sesi (ditulis saat TX selesai)
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
from collections import deque

# ─── KONFIGURASI (harus SAMA dengan rx_ppg_only.py, kecuali PORT) ───
PORT           = 'COM11'          # ganti sesuai port ESP32 TX di laptop ini
BAUD_RATE      = 115200
FS             = 125
PPG_FILE       = 'p000188_ppg.npy'
SEGMEN_DIPILIH = [0, 20, 25]      # hanya segmen (baris) ini yang ditransmisikan
# ──────────────────────────────────────────────────────────────

OUTPUT_DIR = 'hasil_ppg_tx'
WINDOW_S   = 10

# Interval cetak jumlah paket TX ke console (detik)
TX_PRINT_INTERVAL_S = 1.0
TX_CSV_FLUSH_EVERY  = 25
THROUGHPUT_WIN_S    = 3

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PPG_PATH    = os.path.join(SCRIPT_DIR, PPG_FILE)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_PATH, exist_ok=True)

WINDOW_N = WINDOW_S * FS

buf_tx   = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
lock     = threading.Lock()
stop_evt = threading.Event()

stats = {
    'tx_pkt': 0, 'bytes_tx': 0,
    'thr_tx_bps': 0.0, 'thr_tx_kbps': 0.0, 'pkt_rate_tx': 0.0,
    'selesai': False,
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


def load_ppg(filepath, segmen_dipilih=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    print(f"[PPG] Loading '{filepath}'...")
    data = np.load(filepath, allow_pickle=True)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_segmen_total = data.shape[0]
    print(f"[PPG] File berisi {n_segmen_total} segmen, panjang/segmen={data.shape[1]}")

    if segmen_dipilih is not None:
        idx_valid = [i for i in segmen_dipilih if 0 <= i < n_segmen_total]
        idx_invalid = [i for i in segmen_dipilih if i not in idx_valid]
        if idx_invalid:
            print(f"[PPG] PERINGATAN: index segmen {idx_invalid} di luar jangkauan, dilewati.")
        if not idx_valid:
            raise ValueError(f"[PPG] Tidak ada segmen valid dari {segmen_dipilih}")
        data = data[idx_valid]
        print(f"[PPG] Hanya menggunakan segmen: {idx_valid}")

    sig = data.flatten().astype(float)
    print(f"[PPG] {len(sig):,} sampel @ {FS} Hz siap ({len(sig)/FS:.1f} detik | {data.shape[0]} segmen)")
    return sig


# ─── RINGKASAN CSV ───────────────────────────────────────────
def simpan_ringkasan(seq_terkirim, n_target, elapsed_total_s, snapshot):
    avg_kbps     = (snapshot['bytes_tx'] * 8 / elapsed_total_s / 1000.0) if elapsed_total_s > 0 else 0.0
    avg_pkt_rate = (seq_terkirim / elapsed_total_s) if elapsed_total_s > 0 else 0.0
    persen_kirim = (seq_terkirim / n_target * 100) if n_target > 0 else 0.0

    ring_path = os.path.join(OUTPUT_PATH, 'ringkasan_tx.csv')
    with open(ring_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metrik', 'nilai'])
        w.writerow(['port', PORT])
        w.writerow(['sumber_file', PPG_FILE])
        w.writerow(['segmen_dipilih', SEGMEN_DIPILIH])
        w.writerow(['fs_hz', FS])
        w.writerow(['target_paket', n_target])
        w.writerow(['paket_terkirim', seq_terkirim])
        w.writerow(['persen_terkirim', round(persen_kirim, 2)])
        w.writerow(['durasi_sesi_s', round(elapsed_total_s, 3)])
        w.writerow(['bytes_terkirim', snapshot['bytes_tx']])
        w.writerow(['throughput_rata2_kbps', round(avg_kbps, 3)])
        w.writerow(['laju_paket_rata2_pps', round(avg_pkt_rate, 2)])

    print(f"[TX] Ringkasan -> {ring_path}")


# ─── THREAD TX ───────────────────────────────────────────────
def thread_tx():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
        time.sleep(1.5)
        print(f"[TX] Terhubung ke {PORT}")
    except Exception as e:
        print(f"[TX] ERROR buka port: {e}")
        stop_evt.set()
        return

    csv_tx_path = os.path.join(OUTPUT_PATH, 'tx_data.csv')
    f_tx = open(csv_tx_path, 'w', newline='', encoding='utf-8')
    w_tx = csv.writer(f_tx)
    w_tx.writerow([
        'timestamp_s', 'seq', 'ppg_raw', 'ppg_val',
        'bytes_tx', 'bytes_tx_total',
        'thr_tx_bps', 'thr_tx_kbps', 'pkt_rate_tx'
    ])
    t_start_tx = time.time()
    bytes_tx_total = 0
    rows_since_flush = 0

    sig      = load_ppg(PPG_PATH, segmen_dipilih=SEGMEN_DIPILIH)
    n_total  = len(sig)
    src_idx  = 0
    seq      = 0
    interval = 1.0 / FS
    next_t   = time.perf_counter()
    last_print_t = time.perf_counter()

    print(f"[TX] Target total paket: {n_total:,} (harus sama dgn target di rx_ppg_only.py)")

    while not stop_evt.is_set() and src_idx < n_total:
        val = float(sig[src_idx])
        src_idx += 1

        raw    = int(round(val * 10000))
        packet = f"START|PPG:{raw}|SEQ:{seq}|END\n"
        pkt_bytes = len(packet.encode('utf-8'))

        try:
            ser.write(packet.encode('utf-8'))
        except Exception as e:
            print(f"[TX] Write error: {e}")
            break

        meter_tx.update(pkt_bytes)
        bps_tx, pps_tx = meter_tx.get()

        with lock:
            buf_tx.append(val)
            stats['tx_pkt']      = seq + 1
            stats['bytes_tx']   += pkt_bytes
            stats['thr_tx_bps']  = bps_tx
            stats['thr_tx_kbps'] = bps_tx / 1000.0
            stats['pkt_rate_tx'] = pps_tx

        bytes_tx_total += pkt_bytes
        w_tx.writerow([
            round(time.time() - t_start_tx, 3), seq, raw, round(val, 4),
            pkt_bytes, bytes_tx_total,
            round(bps_tx, 1), round(bps_tx / 1000.0, 3), round(pps_tx, 2)
        ])
        rows_since_flush += 1
        if rows_since_flush >= TX_CSV_FLUSH_EVERY:
            f_tx.flush()
            rows_since_flush = 0

        now_t = time.perf_counter()
        if now_t - last_print_t >= TX_PRINT_INTERVAL_S:
            print(f"[TX] Paket terkirim: {seq + 1:,} / {n_total:,} "
                  f"({(seq + 1) / n_total * 100:5.1f}%) | "
                  f"Rate: {pps_tx:5.1f} pkt/s | Thr: {bps_tx/1000:6.2f} kbps")
            last_print_t = now_t

        seq    += 1
        next_t += interval
        sleep   = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

    f_tx.flush()
    f_tx.close()
    ser.close()

    elapsed_total_s = time.time() - t_start_tx
    with lock:
        snapshot = dict(stats)
        stats['selesai'] = True

    print(f"[TX] Selesai — {seq:,} paket terkirim (semua data habis, tidak looping).")
    print(f"[TX] CSV disimpan -> {csv_tx_path}")

    simpan_ringkasan(seq, n_total, elapsed_total_s, snapshot)

    time.sleep(2.0)   # jeda sebentar biar sempat kebaca RX sebelum dashboard nutup
    stop_evt.set()


# ─── DASHBOARD (1 panel: sinyal TX + statistik) ───────────────
def run_dashboard():
    plt.rcParams.update({
        'figure.facecolor': '#0D1117', 'axes.facecolor': '#161B22',
        'axes.edgecolor': '#30363D',   'axes.labelcolor': '#C9D1D9',
        'xtick.color': '#8B949E',      'ytick.color': '#8B949E',
        'grid.color': '#21262D',       'text.color': '#C9D1D9',
        'font.family': 'monospace',
    })

    t_axis = np.linspace(0, WINDOW_S, WINDOW_N)
    fig = plt.figure(figsize=(11, 7))
    sup_title = fig.suptitle(f'PPG Monitor — TX SAJA  |  Port: {PORT}',
                              color='#58A6FF', fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                            left=0.09, right=0.96, top=0.88, bottom=0.08,
                            height_ratios=[1.4, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    ax1.set_title('Sinyal PPG — TX (yang dikirim)', color='#58A6FF', fontsize=10)
    ax1.set_xlim(0, WINDOW_S)
    ax1.set_ylim(-0.1, 4.5)
    ax1.set_xlabel('Waktu (detik)', fontsize=9)
    ax1.set_ylabel('Amplitudo', fontsize=9)
    ax1.grid(True, alpha=0.3)

    line_tx, = ax1.plot(t_axis, list(buf_tx), color='#58A6FF', lw=1.0)

    stats_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                          fontsize=10.5, verticalalignment='top',
                          fontfamily='monospace', color='#E3B341',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22', alpha=0.85))

    def update(_frame):
        if stop_evt.is_set():
            try:
                ani.event_source.stop()
            except Exception:
                pass
            plt.close(fig)
            return line_tx, sup_title, stats_text

        with lock:
            tx = np.array(buf_tx)
            s = dict(stats)

        line_tx.set_ydata(tx)
        if np.ptp(tx) > 0:
            margin = np.ptp(tx) * 0.12
            ax1.set_ylim(np.min(tx) - margin, np.max(tx) + margin)

        sup_title.set_text(
            f'PPG Monitor — TX SAJA  |  Port: {PORT}'
            + ('   ✅ SELESAI' if s['selesai'] else '')
        )

        txt = (
            f"TX paket   : {s['tx_pkt']:>8,}\n"
            f"Throughput : {s['thr_tx_kbps']:>7.3f} kbps\n"
            f"Laju paket : {s['pkt_rate_tx']:>7.1f} pkt/s\n"
            f"Byte TX    : {s['bytes_tx']/1024:>7.1f} KB\n"
            f"Status     : {'SELESAI - menunggu dashboard tertutup otomatis' if s['selesai'] else 'MENGIRIM...'}"
        )
        stats_text.set_text(txt)
        fig.canvas.draw_idle()
        return line_tx, sup_title, stats_text

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

    def on_close(event):
        try:
            ani.event_source.stop()
        except Exception:
            pass
        stop_evt.set()

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()


# ─── MAIN ────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('  PPG Serial Monitor — TX SAJA (proses/laptop terpisah)')
    print('=' * 70)
    print(f'  Port      : {PORT}')
    print(f'  Sumber    : {PPG_PATH}')
    print(f'  Segmen    : {SEGMEN_DIPILIH}')
    print(f'  Output    : {OUTPUT_PATH}/  (tx_data.csv, ringkasan_tx.csv)')
    print('=' * 70)
    print('\nDashboard akan tertutup otomatis setelah selesai mengirim.\n')

    if not os.path.exists(PPG_PATH):
        print(f"❌ ERROR: File PPG tidak ditemukan!\n   Path: {PPG_PATH}")
        return

    t = threading.Thread(target=thread_tx, daemon=True)
    t.start()

    try:
        run_dashboard()
    except KeyboardInterrupt:
        print('\n[Main] Dihentikan.')
        stop_evt.set()

    t.join(timeout=3)
    print('[Main] Selesai.')


if __name__ == '__main__':
    main()