"""
CATATAN: SNR (perbandingan sinyal TX vs RX) DIHAPUS di versi split ini,
karena sebelumnya dihitung dengan membandingkan buffer TX dan RX yang
sama-sama ada di memori satu proses — itu tidak bisa dilakukan lagi kalau
TX dan RX berada di laptop yang berbeda tanpa koneksi jaringan tambahan.
SBP/DBP "TX" tetap akurat karena nilainya sudah ikut dikirim di dalam
setiap paket radio (field SBP:/DBP:), jadi tidak butuh akses ke proses TX.

FIX (segmen aktif macet di 0): field 'SEG:' di paket yang diterima RX
kemungkinan tidak diteruskan utuh oleh firmware ESP32 (struct TDMPacket
tidak selalu bawa field ini lewat radio), jadi fields.get('SEG', 0) selalu
jatuh ke default 0. Sekarang segmen aktif dihitung SENDIRI oleh RX dari
nomor SEQ (sama persis seperti cara TX melabeli segmennya sendiri lewat
seg_boundaries), jadi TIDAK bergantung lagi pada field SEG dari paket.
Syaratnya cuma satu: SEGMEN_DIPILIH, SEG_DURATION_S, FS di sini HARUS
identik dengan tx_only.py (memang sudah jadi syarat skrip ini dari awal,
karena dipakai juga untuk membangun sinyal referensi SNR).
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

# ─── KONFIGURASI (harus SAMA dengan tx_only.py, kecuali PORT) ───
PORT           = 'COM14'          # ganti sesuai port ESP32 RX di laptop ini
BAUD_RATE      = 115200
FS             = 125
ABP_FILE       = 'p000188_abp.npy'   # dipakai sebagai REFERENSI untuk hitung SNR & segmen
SEGMEN_DIPILIH = [0, 20, 25]         # HARUS sama dgn tx_only.py (TOTAL_TARGET + referensi SNR/segmen)
SEG_DURATION_S = 30
# ──────────────────────────────────────────────────────────────

OUTPUT_DIR   = 'hasil_abp_rx'
PEAK_DIST    = 40
PEAK_PROM    = 3
TREND_LEN    = 60
WINDOW_S     = 10
THROUGHPUT_WIN_S = 3
IDLE_TIMEOUT_S = 6.0   # kalau tidak ada paket masuk selama ini, anggap TX sudah selesai
SNR_WINDOW_N = 625     # jumlah sampel terakhir yang dipakai untuk hitung SNR bergerak (5 detik @125Hz)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABP_PATH = os.path.join(SCRIPT_DIR, ABP_FILE)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_PATH, exist_ok=True)

WINDOW_N = WINDOW_S * FS
TOTAL_TARGET = SEG_DURATION_S * FS * len(SEGMEN_DIPILIH)

buf_rx   = deque([0.0] * WINDOW_N, maxlen=WINDOW_N)
sbp_hist = deque([0.0] * TREND_LEN, maxlen=TREND_LEN)
dbp_hist = deque([0.0] * TREND_LEN, maxlen=TREND_LEN)
snr_expected_hist = deque(maxlen=SNR_WINDOW_N)  # nilai REFERENSI (seharusnya diterima), by seq
snr_actual_hist   = deque(maxlen=SNR_WINDOW_N)  # nilai yang BENAR-BENAR diterima
lock     = threading.Lock()
stop_evt = threading.Event()

stats = {
    'rx_pkt': 0,
    'rx_valid': 0,
    'rx_invalid': 0,
    'loss_pkt': 0,
    'loss_pct': 0.0,
    'success_ratio': 0.0,
    'snr_db': 0.0,
    'segment_rx': -1,
    'sbp_tx': 0.0,
    'dbp_tx': 0.0,
    'sbp_rx': 0.0,
    'dbp_rx': 0.0,
    'map_rx': 0.0,
    'pp_rx': 0.0,
    'bytes_rx': 0,
    'thr_kbps': 0.0,
    'pkt_rate': 0.0,
    'selesai': False,
}


def load_abp(filepath, segmen_dipilih=None, seg_duration_s=None, fs=FS):
    """Sama persis dengan versi di tx_only.py — dipakai untuk membangun ulang
    sinyal REFERENSI yang seharusnya diterima (untuk hitung SNR), TANPA perlu
    akses ke proses TX yang jalan di laptop lain.
    Sekarang juga mengembalikan seg_boundaries & seg_labels supaya RX bisa
    menghitung SENDIRI segmen aktif dari nomor SEQ (lihat thread_rx)."""
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
    seg_labels     = idx_valid                    # nomor segmen asli, mis. [0, 20, 25]
    seg_boundaries = np.cumsum(seg_lengths)        # index akhir kumulatif tiap segmen

    sig = np.concatenate(segments)
    return sig, seg_boundaries, seg_labels


def cari_segmen_dari_seq(seq, seg_boundaries, seg_labels):
    """Tentukan segmen aktif HANYA dari nomor SEQ + konfigurasi lokal RX,
    tanpa bergantung pada field SEG di paket (yang ternyata tidak selalu
    diteruskan utuh oleh firmware ESP32)."""
    pos = int(np.searchsorted(seg_boundaries, seq, side='right'))
    pos = min(max(pos, 0), len(seg_labels) - 1)
    return seg_labels[pos]


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


class ABPDetector:
    def __init__(self, fs=FS, win_s=5):
        self.fs = fs
        self.buf = deque(maxlen=win_s * fs)
        self.sbp = self.dbp = self.map_ = self.pp = 0.0
        self.n = 0
        self.refractory = int(fs * 0.4)

    def update(self, val):
        self.buf.append(val)
        self.n += 1
        if self.n % (self.fs // 2) != 0 or len(self.buf) < self.refractory * 2:
            return self.sbp, self.dbp, self.map_, self.pp
        arr = np.array(self.buf)
        peaks, _ = find_peaks(arr, distance=PEAK_DIST, prominence=PEAK_PROM)
        valleys, _ = find_peaks(-arr, distance=PEAK_DIST, prominence=PEAK_PROM)
        if len(peaks) > 0:
            self.sbp = float(arr[peaks].mean())
        if len(valleys) > 0:
            self.dbp = float(arr[valleys].mean())
        if self.sbp > 0 and self.dbp > 0:
            self.pp = self.sbp - self.dbp
            self.map_ = self.dbp + self.pp / 3.0
        return self.sbp, self.dbp, self.map_, self.pp


# ─── THREAD RX ───────────────────────────────────────────────
def thread_rx():
    try:
        ser_rx = serial.Serial(PORT, BAUD_RATE, timeout=0.5)
        time.sleep(1.5)
        print(f"[RX] Terhubung ke {PORT}")
    except Exception as e:
        print(f"[RX] ERROR: {e}")
        stop_evt.set()
        return

    # Muat sinyal REFERENSI (harus identik dengan yang di-generate tx_only.py)
    # supaya SNR & SEGMEN AKTIF bisa dihitung sendiri di sisi RX, tanpa akses
    # ke proses TX ataupun bergantung pada field SEG dari paket.
    try:
        sig_ref, seg_boundaries_ref, seg_labels_ref = load_abp(
            ABP_PATH, segmen_dipilih=SEGMEN_DIPILIH, seg_duration_s=SEG_DURATION_S, fs=FS
        )
        n_ref = len(sig_ref)
        print(f"[RX] Sinyal referensi SNR/segmen dimuat: {n_ref:,} sampel (dari {ABP_PATH})")
        if n_ref != TOTAL_TARGET:
            print(f"[RX] ⚠️  PERINGATAN: panjang referensi ({n_ref:,}) != TOTAL_TARGET ({TOTAL_TARGET:,})."
                  f" Cek konfigurasi SEGMEN_DIPILIH/SEG_DURATION_S sama dengan tx_only.py!")
    except Exception as e:
        print(f"[RX] ⚠️  Tidak bisa muat sinyal referensi ({e}) — SNR & segmen aktif tidak akan dihitung.")
        sig_ref = None
        n_ref = 0
        seg_boundaries_ref = None
        seg_labels_ref = None

    detector = ABPDetector()
    csv_path = os.path.join(OUTPUT_PATH, 'rx_data.csv')
    csv_f = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_f)
    writer.writerow([
        'elapsed_s', 'seq', 'segment', 'abp_raw', 'abp_mmhg',
        'sbp_rx', 'dbp_rx', 'map_rx', 'pp_rx',
        'sbp_tx', 'dbp_tx', 'selisih_sbp', 'selisih_dbp',
        'loss_pct', 'success_ratio', 'snr_db'
    ])

    print(f"[RX] Target total paket (dari konfigurasi): {TOTAL_TARGET:,}")
    print(f"[RX] Menunggu data... (auto-berhenti kalau tidak ada paket masuk {IDLE_TIMEOUT_S:.0f} detik)")

    t_start = time.time()
    buf = b""
    rx_valid_count = 0
    last_rx_time = time.time()
    started_receiving = False

    while not stop_evt.is_set():
        # auto-stop kalau sempat terima data lalu diam cukup lama (TX selesai/berhenti)
        if started_receiving and (time.time() - last_rx_time > IDLE_TIMEOUT_S):
            print(f"\n[RX] Tidak ada data baru selama {IDLE_TIMEOUT_S:.0f} detik — anggap TX sudah selesai.")
            with lock:
                stats['selesai'] = True
            break

        try:
            if ser_rx.in_waiting > 0:
                chunk = ser_rx.read(ser_rx.in_waiting)
            else:
                time.sleep(0.001)
                continue
        except serial.SerialException as e:
            print(f"[RX] Read error: {e}")
            break

        if not chunk:
            continue

        buf += chunk

        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue

            try:
                text = line.decode('utf-8', errors='replace')

                if not (text.startswith('START') and 'END' in text):
                    with lock:
                        stats['rx_invalid'] += 1
                        stats['rx_pkt'] += 1
                    continue

                fields = {}
                for part in text.replace('START|', '').replace('|END', '').split('|'):
                    if ':' in part:
                        k, v = part.split(':', 1)
                        fields[k] = v

                elapsed_s = round(time.time() - t_start, 3)
                is_valid = True

                if not all(k in fields for k in ('ABP', 'SEQ')):
                    is_valid = False
                    with lock:
                        stats['rx_invalid'] += 1
                        stats['rx_pkt'] += 1
                    continue

                last_rx_time = time.time()
                started_receiving = True

                raw = int(fields['ABP'])
                seq = int(fields['SEQ'])

                # FIX: segmen aktif dihitung dari SEQ + referensi lokal,
                # BUKAN dari fields.get('SEG', 0) — field itu ternyata tidak
                # selalu ikut diteruskan lewat radio oleh firmware ESP32,
                # jadi kalau diandalkan akan selalu kebaca 0.
                if seg_boundaries_ref is not None:
                    seg_idx = cari_segmen_dari_seq(seq, seg_boundaries_ref, seg_labels_ref)
                else:
                    seg_idx = int(fields.get('SEG', 0))  # fallback kalau referensi gagal dimuat

                sbp_tx_now = float(fields.get('SBP', 0))
                dbp_tx_now = float(fields.get('DBP', 0))
                val_mm = raw / 100.0

                if not (2000 <= raw <= 25000):
                    is_valid = False

                with lock:
                    stats['rx_pkt'] += 1
                    stats['segment_rx'] = seg_idx  # update begitu paket ke-parse, tidak nunggu is_valid

                    if is_valid:
                        rx_valid_count += 1
                        stats['rx_valid'] = rx_valid_count
                    else:
                        stats['rx_invalid'] += 1

                    # FIX: loss/success ratio dihitung dari TOTAL_TARGET
                    # (konstanta hasil konfigurasi), BUKAN dari stats TX
                    # live seperti versi 1-proses — karena proses TX ada
                    # di laptop lain, tidak bisa diakses langsung.
                    stats['loss_pkt'] = max(0, TOTAL_TARGET - rx_valid_count)
                    stats['loss_pct'] = (stats['loss_pkt'] / TOTAL_TARGET * 100.0) if TOTAL_TARGET > 0 else 0.0
                    stats['success_ratio'] = (rx_valid_count / TOTAL_TARGET * 100.0) if TOTAL_TARGET > 0 else 0.0

                    stats['bytes_rx'] += len(line) + 1

                if is_valid:
                    n_bytes_line = len(line) + 1
                    meter.update(n_bytes_line, n_pkts=1)
                    bps, pps = meter.get()

                    with lock:
                        stats['thr_kbps'] = bps / 1000.0
                        stats['pkt_rate'] = pps

                    sbp_rx, dbp_rx, map_rx, pp_rx = detector.update(val_mm)

                    if sbp_rx > 0 and dbp_rx > 0:
                        with lock:
                            sbp_hist.append(sbp_rx)
                            dbp_hist.append(dbp_rx)
                            stats['sbp_rx'] = sbp_rx
                            stats['dbp_rx'] = dbp_rx
                            stats['map_rx'] = map_rx
                            stats['pp_rx'] = pp_rx
                            stats['sbp_tx'] = sbp_tx_now
                            stats['dbp_tx'] = dbp_tx_now

                    with lock:
                        buf_rx.append(val_mm)

                    # SNR dihitung SENDIRI di sisi RX, real-time, tanpa akses
                    # ke proses TX — dengan membandingkan nilai yang diterima
                    # ke sinyal REFERENSI lokal (sig_ref) pada index seq yang
                    # sama (urutan TX 100% deterministik: seq ke-N = sig[N]).
                    if sig_ref is not None and 0 <= seq < n_ref:
                        expected_val = float(sig_ref[seq])
                        snr_expected_hist.append(expected_val)
                        snr_actual_hist.append(val_mm)
                        if len(snr_expected_hist) >= 10:
                            exp_arr = np.array(snr_expected_hist)
                            act_arr = np.array(snr_actual_hist)
                            ps = np.mean(exp_arr ** 2)
                            pn = np.mean((exp_arr - act_arr) ** 2)
                            snr_now = 10 * np.log10(ps / pn) if pn > 1e-12 else 99.0
                            with lock:
                                stats['snr_db'] = snr_now

                    sel_sbp = round(sbp_rx - sbp_tx_now, 2) if sbp_rx > 0 and sbp_tx_now > 0 else ''
                    sel_dbp = round(dbp_rx - dbp_tx_now, 2) if dbp_rx > 0 and dbp_tx_now > 0 else ''
                else:
                    sel_sbp = ''
                    sel_dbp = ''
                    sbp_rx = dbp_rx = map_rx = pp_rx = 0.0

                with lock:
                    loss_pct_now = stats['loss_pct']
                    success_ratio_now = stats['success_ratio']
                    snr_db_now = stats['snr_db']

                writer.writerow([
                    elapsed_s, seq, seg_idx, raw, round(val_mm, 2) if is_valid else '',
                    round(sbp_rx, 1) if sbp_rx else '',
                    round(dbp_rx, 1) if dbp_rx else '',
                    round(map_rx, 1) if map_rx else '',
                    round(pp_rx, 1) if pp_rx else '',
                    round(sbp_tx_now, 1), round(dbp_tx_now, 1),
                    sel_sbp, sel_dbp,
                    round(loss_pct_now, 2), round(success_ratio_now, 2),
                    round(snr_db_now, 2),
                ])
                csv_f.flush()

            except Exception:
                with lock:
                    stats['rx_invalid'] += 1
                    stats['rx_pkt'] += 1
                continue

    csv_f.close()
    ser_rx.close()
    with lock:
        stats['selesai'] = True
    print(f"\n[RX] Selesai — {stats['rx_pkt']:,} paket diterima, {stats['rx_valid']:,} valid")
    print(f"[RX] Data RX → {csv_path}")
    time.sleep(2.0)
    stop_evt.set()


# ─── RINGKASAN ───────────────────────────────────────────────
def cetak_ringkasan():
    with lock:
        s = dict(stats)

    print('\n' + '=' * 70)
    print('  RINGKASAN AKHIR — ABP Monitor (RX SAJA)')
    print('=' * 70)
    print(f"  Segmen dikonfigurasi : {SEGMEN_DIPILIH}")
    print(f"  Target total paket   : {TOTAL_TARGET:,}")
    print(f"  Paket RX diterima    : {s['rx_pkt']:,}")
    print(f"  Paket RX valid       : {s['rx_valid']:,}")
    print(f"  Paket RX invalid     : {s['rx_invalid']:,}")
    print(f"  Paket loss           : {s['loss_pkt']:,}")
    print(f"  Loss rate            : {s['loss_pct']:.2f}%")
    print(f"  Success Ratio        : {s['success_ratio']:.2f}%")
    print(f"  SNR (vs referensi)   : {s['snr_db']:.2f} dB")
    print(f"\n  Throughput           : {s['thr_kbps']:.3f} kbps")
    print(f"  Laju paket           : {s['pkt_rate']:.1f} pkt/s")
    print(f"  SBP RX / TX (terakhir): {s['sbp_rx']:.1f} / {s['sbp_tx']:.1f} mmHg")
    print(f"  DBP RX / TX (terakhir): {s['dbp_rx']:.1f} / {s['dbp_tx']:.1f} mmHg")
    print('=' * 70)

    ring_path = os.path.join(OUTPUT_PATH, 'ringkasan.csv')
    with open(ring_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metrik', 'nilai'])
        w.writerow(['segmen_dipilih', '|'.join(str(i) for i in SEGMEN_DIPILIH)])
        w.writerow(['total_target_paket', TOTAL_TARGET])
        w.writerow(['paket_rx_total', s['rx_pkt']])
        w.writerow(['paket_rx_valid', s['rx_valid']])
        w.writerow(['paket_rx_invalid', s['rx_invalid']])
        w.writerow(['loss_pkt', s['loss_pkt']])
        w.writerow(['loss_pct', round(s['loss_pct'], 2)])
        w.writerow(['success_ratio', round(s['success_ratio'], 2)])
        w.writerow(['snr_db', round(s['snr_db'], 2)])
        w.writerow(['thr_kbps', round(s['thr_kbps'], 3)])
        w.writerow(['pkt_rate', round(s['pkt_rate'], 1)])
    print(f"\n[✓] Ringkasan → {ring_path}\n")


# ─── DASHBOARD (1 panel: sinyal RX + statistik) ───────────────
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
        f'ABP Monitor — RX SAJA  |  Port: {PORT}\nSegmen aktif: --',
        color='#3FB950', fontsize=13, fontweight='bold'
    )

    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                           left=0.09, right=0.96, top=0.87, bottom=0.08,
                           height_ratios=[1.4, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    ax1.set_title('Sinyal ABP - RX (yang diterima)', color='#3FB950', fontsize=10)
    ax1.set_xlim(0, WINDOW_S)
    ax1.set_ylim(30, 175)
    ax1.set_xlabel('Waktu (detik)', fontsize=9)
    ax1.set_ylabel('mmHg', fontsize=9)
    ax1.grid(True, alpha=0.3)

    line_rx, = ax1.plot(t_axis, list(buf_rx), color='#3FB950', lw=1.0)

    lbl_sbp = ax1.text(0.985, 0.93, 'SBP: -- mmHg', transform=ax1.transAxes,
                       fontsize=11, fontweight='bold', color='#F4C275',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117', alpha=0.85))
    lbl_dbp = ax1.text(0.985, 0.75, 'DBP: -- mmHg', transform=ax1.transAxes,
                       fontsize=11, fontweight='bold', color='#5DCAA5',
                       ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1117', alpha=0.85))

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
            return line_rx, lbl_sbp, lbl_dbp, sup_title, stats_text

        with lock:
            rx = np.array(buf_rx)
            s = dict(stats)

        line_rx.set_ydata(rx)
        if np.ptp(rx) > 0:
            margin = np.ptp(rx) * 0.12
            ax1.set_ylim(np.min(rx) - margin, np.max(rx) + margin)

        lbl_sbp.set_text(f"SBP: {s['sbp_rx']:.0f} mmHg" if s['sbp_rx'] > 0 else 'SBP: -- mmHg')
        lbl_dbp.set_text(f"DBP: {s['dbp_rx']:.0f} mmHg" if s['dbp_rx'] > 0 else 'DBP: -- mmHg')

        seg_now = s['segment_rx']
        sup_title.set_text(
            f'ABP Monitor — RX SAJA  |  Port: {PORT}\n'
            f'Segmen aktif: {seg_now if seg_now >= 0 else "--"}'
            + ('   ✅ SELESAI' if s['selesai'] else '')
        )

        C = "  |  "
        txt = (
            f"RX paket  : {s['rx_pkt']:>8,} / target {TOTAL_TARGET:,}{C}"
            f"RX valid  : {s['rx_valid']:>8,}\n"
            f"Loss pkt  : {s['loss_pkt']:>8,}{C}"
            f"Loss %    : {s['loss_pct']:>7.2f} %\n"
            f"Success % : {s['success_ratio']:>7.2f} %\n"
            f"Throughput: {s['thr_kbps']:>7.3f} kbps{C}"
            f"Laju paket: {s['pkt_rate']:>6.1f} pkt/s\n"
            f"SBP TX*   : {s['sbp_tx']:>6.1f} mmHg{C}"
            f"SBP RX    : {s['sbp_rx']:>6.1f} mmHg\n"
            f"DBP TX*   : {s['dbp_tx']:>6.1f} mmHg{C}"
            f"DBP RX    : {s['dbp_rx']:>6.1f} mmHg\n"
            f"(*SBP/DBP TX didapat dari isi paket yang diterima, bukan akses langsung ke proses TX)"
        )
        stats_text.set_text(txt)
        fig.canvas.draw_idle()
        return line_rx, lbl_sbp, lbl_dbp, sup_title, stats_text

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
    print('  ABP Serial Monitor — RX SAJA (setup 2 laptop)')
    print('=' * 70)
    print(f'  Port      : {PORT}')
    print(f'  Segmen    : {SEGMEN_DIPILIH}  (@ {SEG_DURATION_S}s → total {SEG_DURATION_S*len(SEGMEN_DIPILIH)}s)')
    print(f'  Target    : {TOTAL_TARGET:,} paket')
    print(f'  Output    : {OUTPUT_PATH}/')
    print('=' * 70)
    print('\nTutup jendela dashboard untuk berhenti manual (atau otomatis kalau TX selesai).\n')

    t = threading.Thread(target=thread_rx, daemon=True)
    t.start()

    try:
        run_dashboard()
    except KeyboardInterrupt:
        print('\n[Main] Dihentikan.')
        stop_evt.set()

    t.join(timeout=3)
    cetak_ringkasan()
    print('[Main] Selesai.')


if __name__ == '__main__':
    main()