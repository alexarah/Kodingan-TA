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

# ─── KONFIGURASI  ───
PORT           = 'COM11'         
BAUD_RATE      = 115200
FS             = 125
ABP_FILE       = 'p000188_abp.npy'
SEGMEN_DIPILIH = [0, 20, 25]
SEG_DURATION_S = 30               # tiap segmen = 30 detik → total 90 detik
# ──────────────────────────────────────────────────────────────

OUTPUT_DIR   = 'hasil_abp_tx'
PEAK_DIST    = 40
PEAK_PROM    = 3
WINDOW_S     = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABP_PATH = os.path.join(SCRIPT_DIR, ABP_FILE)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_PATH, exist_ok=True)

WINDOW_N = WINDOW_S * FS
TOTAL_TARGET = SEG_DURATION_S * FS * len(SEGMEN_DIPILIH)

buf_tx   = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
lock     = threading.Lock()
stop_evt = threading.Event()

stats = {
    'tx_pkt': 0,
    'segment_tx': -1,
    'sbp_tx': 0.0,
    'dbp_tx': 0.0,
    'map_tx': 0.0,
    'pp_tx': 0.0,
    'bytes_tx': 0,
    'thr_kbps': 0.0,
    'selesai': False,
}


def load_abp(filepath, segmen_dipilih=None, seg_duration_s=None, fs=FS):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    data = np.load(filepath, allow_pickle=True)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if segmen_dipilih is not None:
        idx_valid = [i for i in segmen_dipilih if 0 <= i < data.shape[0]]
        if not idx_valid:
            raise ValueError(f"Tidak ada segmen valid dari {segmen_dipilih}")
    else:
        idx_valid = list(range(data.shape[0]))

    seg_len_target = int(seg_duration_s * fs) if seg_duration_s else None

    segments = []
    for i in idx_valid:
        s = data[i].astype(float)
        if seg_len_target is not None:
            if len(s) >= seg_len_target:
                s = s[:seg_len_target]
            else:
                reps = int(np.ceil(seg_len_target / len(s)))
                s = np.tile(s, reps)[:seg_len_target]
        segments.append(s)

    seg_lengths    = [len(s) for s in segments]
    seg_labels     = idx_valid
    seg_boundaries = np.cumsum(seg_lengths)

    sig = np.concatenate(segments)
    return sig, seg_boundaries, seg_labels


def hitung_sbp_dbp(buf_list):
    if len(buf_list) < PEAK_DIST * 2:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(buf_list, dtype=float)
    peaks, _ = find_peaks(arr, distance=PEAK_DIST, prominence=PEAK_PROM)
    valleys, _ = find_peaks(-arr, distance=PEAK_DIST, prominence=PEAK_PROM)
    sbp = float(arr[peaks].mean()) if len(peaks) > 0 else 0.0
    dbp = float(arr[valleys].mean()) if len(valleys) > 0 else 0.0
    pp = sbp - dbp if sbp > 0 and dbp > 0 else 0.0
    map_ = dbp + pp / 3.0 if pp > 0 else 0.0
    return sbp, dbp, map_, pp


# ─── THREAD TX ───────────────────────────────────────────────
def thread_tx():
    try:
        ser_tx = serial.Serial(PORT, BAUD_RATE, timeout=2)
        time.sleep(1.5)
        print(f"[TX] Terhubung ke {PORT}")
    except Exception as e:
        print(f"[TX] ERROR: {e}")
        stop_evt.set()
        return

    sig, seg_boundaries, seg_labels = load_abp(
        ABP_PATH, segmen_dipilih=SEGMEN_DIPILIH, seg_duration_s=SEG_DURATION_S, fs=FS
    )
    n_total = len(sig)
    src_idx = 0
    seq = 0
    buf_win = []
    sbp_val = dbp_val = map_val = pp_val = 0.0
    interval = 1.0 / FS
    next_t = time.perf_counter()
    t_start = time.time()

    csv_path_tx = os.path.join(OUTPUT_PATH, 'tx_data.csv')
    csv_f_tx = open(csv_path_tx, 'w', newline='', encoding='utf-8')
    writer_tx = csv.writer(csv_f_tx)
    writer_tx.writerow([
        'elapsed_s', 'seq', 'segment', 'abp_raw', 'abp_mmhg',
        'sbp_tx', 'dbp_tx', 'map_tx', 'pp_tx'
    ])

    print(f"[TX] Target total paket: {n_total:,} (harus sama dgn TOTAL_TARGET di rx_only.py = {TOTAL_TARGET:,})")

    while not stop_evt.is_set() and src_idx < n_total:
        val = float(sig[src_idx])

        pos_segmen = int(np.searchsorted(seg_boundaries, src_idx, side='right'))
        pos_segmen = min(pos_segmen, len(seg_labels) - 1)
        seg_idx = seg_labels[pos_segmen]

        src_idx += 1

        raw = int(round(val * 100))
        val_mmhg = raw / 100.0

        buf_win.append(val)
        if len(buf_win) > FS * 5:
            buf_win.pop(0)
        if seq % FS == 0:
            sbp_val, dbp_val, map_val, pp_val = hitung_sbp_dbp(buf_win)

        packet = (f"START|ABP:{raw}|SBP:{sbp_val:.1f}|DBP:{dbp_val:.1f}"
                  f"|MAP:{map_val:.1f}|PP:{pp_val:.1f}|SEQ:{seq}|SEG:{seg_idx}|END\n")
        pkt_bytes = len(packet.encode('utf-8'))

        try:
            ser_tx.write(packet.encode('utf-8'))
            ser_tx.flush()
        except Exception as e:
            print(f"[TX] Write error: {e}")
            break

        elapsed_s = round(time.time() - t_start, 4)

        with lock:
            buf_tx.append(val_mmhg)
            stats['tx_pkt'] = seq + 1
            stats['sbp_tx'] = sbp_val
            stats['dbp_tx'] = dbp_val
            stats['map_tx'] = map_val
            stats['pp_tx'] = pp_val
            stats['segment_tx'] = seg_idx
            stats['bytes_tx'] += pkt_bytes
            elapsed_now = time.time() - t_start
            stats['thr_kbps'] = (stats['bytes_tx'] * 8 / elapsed_now / 1000.0) if elapsed_now > 0 else 0.0

        writer_tx.writerow([
            elapsed_s, seq, seg_idx, raw, round(val_mmhg, 2),
            round(sbp_val, 1), round(dbp_val, 1), round(map_val, 1), round(pp_val, 1)
        ])
        csv_f_tx.flush()

        seq += 1
        next_t += interval
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)

    csv_f_tx.close()
    ser_tx.close()
    print(f"\n[TX] Selesai — {seq:,} paket terkirim (target {n_total:,}).")
    print(f"[TX] Data TX → {csv_path_tx}")

    with lock:
        stats['selesai'] = True

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
    fig = plt.figure(figsize=(11, 8))
    sup_title = fig.suptitle(
        f'ABP Monitor — TX SAJA  |  Port: {PORT}\nSegmen aktif: --',
        color='#58A6FF', fontsize=13, fontweight='bold'
    )

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

        seg_now = s['segment_tx']
        sup_title.set_text(
            f'ABP Monitor — TX SAJA  |  Port: {PORT}\n'
            f'Segmen aktif: {seg_now if seg_now >= 0 else "--"}'
            + ('   SELESAI' if s['selesai'] else '')
        )

        progress_pct = (s['tx_pkt'] / TOTAL_TARGET * 100) if TOTAL_TARGET > 0 else 0
        txt = (
            f"TX paket   : {s['tx_pkt']:>8,} / {TOTAL_TARGET:,}  ({progress_pct:5.1f} %)\n"
            f"Throughput : {s['thr_kbps']:>7.3f} kbps\n"
            f"SBP TX     : {s['sbp_tx']:>6.1f} mmHg   DBP TX : {s['dbp_tx']:>6.1f} mmHg\n"
            f"MAP TX     : {s['map_tx']:>6.1f} mmHg   PP  TX : {s['pp_tx']:>6.1f} mmHg\n"
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
    print('  ABP Serial Monitor — TX SAJA (setup 2 laptop)')
    print('=' * 70)
    print(f'  Port      : {PORT}')
    print(f'  Sumber    : {ABP_PATH}')
    print(f'  Segmen    : {SEGMEN_DIPILIH}  (@ {SEG_DURATION_S}s → total {SEG_DURATION_S*len(SEGMEN_DIPILIH)}s)')
    print(f'  Target    : {TOTAL_TARGET:,} paket')
    print(f'  Output    : {OUTPUT_PATH}/')
    print('=' * 70)
    print('\nDashboard akan tertutup otomatis setelah selesai mengirim.\n')

    if not os.path.exists(ABP_PATH):
        print(f"ERROR: File ABP tidak ditemukan!\n   Path: {ABP_PATH}")
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
