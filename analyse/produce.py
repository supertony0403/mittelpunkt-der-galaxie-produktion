"""Mittelpunkt der Galaxie - full production from demucs stems of v3.

Lead vocal autotune (Graillon 3, C# minor), hook doubles/octaves, own echo adlibs,
vocal-chop hook part, radio intro with producer tag, risers/impacts, beat punch,
mix bus and a true-peak limited master for streaming.

Run:  ~/opt/stemsplit/.venv/bin/python -P produce.py <project-dir> [--lufs -9]
"""
import argparse
import json
import numpy as np
import soundfile as sf
import scipy.signal as ss
import librosa
import pedalboard as pb

GRAILLON = "/usr/lib/vst3/Auburn Sounds Graillon 3.vst3"
SR = 44100
# C# minor / E major
SCALE = {"c": False, "c_sharp": True, "d": False, "d_sharp": True, "e": True, "f": False,
         "f_sharp": True, "g": False, "g_sharp": True, "a": True, "a_sharp": False, "b": True}

# sections (seconds, measured on v3 transcript + RMS)
HOOKS = [(0.0, 30.2), (66.4, 87.1), (165.4, 186.3)]
PRE = [(55.8, 66.1)]
VERSES = [(30.2, 55.8), (87.1, 117.6)]
BRIDGE = (117.6, 165.4)
OUTRO = (186.3, 216.0)
DROPS = [20.86, 66.9, 165.9]          # snapped to beats below


def load(path, sr=SR):
    y, r = sf.read(path, dtype="float32", always_2d=True)
    assert r == sr, (path, r)
    return y.T.copy()                  # (2, n)


def fit(x, n):
    if x.shape[1] >= n:
        return x[:, :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[1])))


def db(g):
    return 10 ** (g / 20)


def region_env(n, regions, fade=0.08, sr=SR):
    """0..1 envelope that is 1 inside regions, with short cosine fades."""
    e = np.zeros(n, np.float32)
    f = int(fade * sr)
    for a, b in regions:
        i, j = int(a * sr), min(n, int(b * sr))
        e[i:j] = 1
        r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))
        lo = max(0, i - f)
        e[lo:i] = np.maximum(e[lo:i], r[f - (i - lo):])
        hi = min(n, j + f)
        e[j:hi] = np.maximum(e[j:hi], r[::-1][:hi - j])
    return e


def pan(x, p):
    """x mono (n,) or stereo -> stereo, p in -1..1 constant power."""
    m = x.mean(0) if x.ndim == 2 else x
    a = (p + 1) * np.pi / 4
    return np.stack([m * np.cos(a), m * np.sin(a)])


def shift(x, sec, sr=SR):
    k = int(round(sec * sr))
    if k == 0:
        return x
    out = np.zeros_like(x)
    if k > 0:
        out[:, k:] = x[:, :-k]
    else:
        out[:, :k] = x[:, -k:]
    return out


CORRECTION = True   # --tune none switches pitch correction off everywhere (shifts stay)


def graillon(x, sr=SR, **kw):
    p = pb.load_plugin(GRAILLON)
    for note, ok in SCALE.items():
        setattr(p, f"allow_{note}", ok)
    p.correction = True
    p.corr_amount = 100.0
    p.dry_mix_db = -np.inf
    p.wet_mix_db = 0.0
    for k, v in kw.items():
        setattr(p, k, v)
    if not CORRECTION:
        p.correction = False
    return p(x.astype(np.float32), sr)


def board(x, *fx, sr=SR):
    return pb.Pedalboard(list(fx))(x.astype(np.float32), sr)


def deess(x, sr=SR, freq=6500, thr_db=-30, ratio=4.0):
    """split-band de-esser: compress only the band above freq."""
    b, a = ss.butter(4, freq / (sr / 2), "high")
    hi = ss.filtfilt(b, a, x, axis=1).astype(np.float32)
    lo = x - hi
    hi_c = board(hi, pb.Compressor(threshold_db=thr_db, ratio=ratio, attack_ms=1, release_ms=60))
    return lo + hi_c


def envelope(x, sr=SR, win=0.02):
    m = np.abs(x).max(0) if x.ndim == 2 else np.abs(x)
    k = int(win * sr)
    return np.sqrt(np.convolve(m ** 2, np.ones(k) / k, "same"))


def noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def riser(dur, sr=SR, seed=1):
    n = int(dur * sr)
    t = np.linspace(0, 1, n)
    x = noise(n, seed)
    out = np.zeros(n, np.float32)
    # time-varying bandpass by blocks
    blk = 256
    zi = None
    for i in range(0, n, blk):
        fc = 300 * (40 ** t[i])             # 300 Hz -> 12 kHz
        sos = ss.butter(2, [fc * 0.7 / (sr / 2), min(0.99, fc * 1.4 / (sr / 2))], "band", output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[i:i + blk], zi = ss.sosfilt(sos, x[i:i + blk], zi=zi)   # carry state -> no block clicks
    out *= (t ** 2.2) / (np.abs(out).max() + 1e-9)
    st = np.stack([out, np.concatenate([np.zeros(300, np.float32), out[:-300]])])   # 7 ms offset = width          # a little width
    return board(st, pb.Reverb(room_size=0.7, wet_level=0.3, dry_level=0.8, width=1.0))


def impact(sr=SR, seed=2):
    n = int(2.6 * sr)
    t = np.arange(n) / sr
    f = 58 * np.exp(-t * 1.2) + 28                    # sub drop 86 -> 28 Hz
    ph = 2 * np.pi * np.cumsum(f) / sr
    sub = np.sin(ph) * np.exp(-t * 1.6)
    burst = noise(n, seed) * np.exp(-t * 18)
    b, a = ss.butter(2, 2500 / (sr / 2))
    burst = ss.lfilter(b, a, burst)
    x = (0.9 * sub + 0.35 * burst).astype(np.float32)
    st = np.stack([x, x])
    return board(st, pb.Reverb(room_size=0.85, wet_level=0.25, dry_level=0.9))


def radio(x, sr=SR, seed=3):
    """telephone/radio transmission treatment + static."""
    y = board(x, pb.HighpassFilter(500), pb.LowpassFilter(3200),
              pb.Distortion(drive_db=14), pb.Compressor(threshold_db=-22, ratio=6),
              pb.HighpassFilter(450), pb.LowpassFilter(3500))
    n = y.shape[1]
    st = noise(n, seed) * 0.05
    crackle = (np.random.default_rng(seed + 1).random(n) > 0.9993) * noise(n, seed + 2) * 0.5
    b, a = ss.butter(2, [800 / (sr / 2), 6000 / (sr / 2)], "band")
    s = ss.lfilter(b, a, st + crackle).astype(np.float32)
    return y + np.stack([s, s])


def softclip(x, ceiling_db):
    c = db(ceiling_db)
    return (c * np.tanh(x / c)).astype(np.float32)


def snap(t, beats):
    return float(beats[np.argmin(np.abs(beats - t))])


def true_peak_limit(x, ceiling_db=-1.0, sr=SR, look_ms=5.0, release_ms=80.0, os_=4):
    """lookahead brickwall limiter operating on 4x oversampled peak estimate."""
    up = ss.resample_poly(x, os_, 1, axis=1)
    pk = np.abs(up).max(0).reshape(-1, os_).max(1)[: x.shape[1]]
    pk = np.pad(pk, (0, x.shape[1] - len(pk)))
    ceil = db(ceiling_db)
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    L = int(look_ms * 1e-3 * sr)
    # running min over lookahead window (future), so gain is down before the peak
    from scipy.ndimage import minimum_filter1d
    g = minimum_filter1d(need, size=2 * L + 1, mode="nearest")
    # smooth: instant attack (already ahead), exponential release
    rel = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    sm = np.empty_like(g)
    cur = 1.0
    for i in range(len(g)):            # numba-free but fine for ~10M samples
        v = g[i]
        cur = v if v < cur else v + (cur - v) * rel
        sm[i] = cur
    # attack smoothing (linear ramp over L) to avoid clicks
    k = np.ones(L) / L
    sm = np.convolve(sm, k, "same")    # centred box (+-L/2) inside the +-L min window -> smooth ramp, still safe
    return x * sm


def lufs(x, sr=SR):
    import pyloudnorm as pyln
    return pyln.Meter(sr).integrated_loudness(x.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proj")
    ap.add_argument("--lufs", type=float, default=-10.0)
    ap.add_argument("--tune", choices=["light", "none"], default="light")
    ap.add_argument("--name", default="Mittelpunkt der Galaxie - Master 24bit.wav")
    args = ap.parse_args()
    P = args.proj
    S = P + "/stems/htdemucs_ft"
    beats = np.load(P + "/analyse/v3_beats.npy")
    bar = 4 * float(np.median(np.diff(beats)))
    drops = [snap(t, beats) for t in DROPS]

    voc = load(S + "/v3/vocals.wav")
    drm = load(S + "/v3/drums.wav")
    bas = load(S + "/v3/bass.wav")
    oth = load(S + "/v3/other.wav")
    N = voc.shape[1] + int(1.5 * SR)
    voc, drm, bas, oth = (fit(a, N) for a in (voc, drm, bas, oth))
    log = {}

    # ---------- LEAD VOCAL ----------
    lead_in = board(voc, pb.HighpassFilter(95), pb.NoiseGate(threshold_db=-52, ratio=3, release_ms=120))
    global CORRECTION
    TUNE = args.tune != "none"
    CORRECTION = TUNE
    lead_t = lead_in if not TUNE else graillon(lead_in, smooth=18.0, inertia=30.0, corr_amount=55.0, snap_max_st=1.5)   # natural: gentle pull, no robot
    lead = board(lead_t,
                 pb.PeakFilter(280, -2.5, 1.0),
                 pb.PeakFilter(3400, 1.5, 0.8),
                 pb.HighShelfFilter(10000, 2.0),
                 pb.Compressor(threshold_db=-20, ratio=4, attack_ms=4, release_ms=70),
                 pb.Distortion(drive_db=2.0))
    lead = deess(lead)
    lead = board(lead, pb.Compressor(threshold_db=-10, ratio=2.5, attack_ms=12, release_ms=120), pb.Gain(2.0))

    # ---------- HOOK STACK ----------
    n = N
    hook_env = region_env(n, HOOKS)
    hook_src = lead_in * hook_env
    dblL = graillon(hook_src, smooth=12.0, inertia=20.0, corr_amount=65.0, pitch_shift_st=0.10)   # soft tuned, +10 ct
    dblR = graillon(hook_src, smooth=12.0, inertia=20.0, corr_amount=65.0, pitch_shift_st=-0.09)
    octd = graillon(hook_src, smooth=12.0, corr_amount=60.0, pitch_shift_st=-12.0, formant=True, formant_amount=100.0)
    octu = graillon(hook_src, smooth=12.0, corr_amount=60.0, pitch_shift_st=12.0, formant=True, formant_amount=100.0)
    dbl_chain = (pb.HighpassFilter(160), pb.PeakFilter(300, -3, 1.0), pb.HighShelfFilter(9000, 2),
                 pb.Compressor(threshold_db=-24, ratio=6, attack_ms=2, release_ms=60))
    dblL = board(shift(dblL, 0.017), *dbl_chain)
    dblR = board(shift(dblR, 0.026), *dbl_chain)
    octd = board(octd, pb.LowpassFilter(5000), pb.Compressor(threshold_db=-22, ratio=4))
    octu = board(octu, pb.HighpassFilter(900), pb.LowpassFilter(12000), pb.Compressor(threshold_db=-24, ratio=5))
    hook = (pan(dblL, -0.85) * db(-7) + pan(dblR, 0.85) * db(-7)
            + pan(octd, 0.0) * db(-11) + pan(octu, -0.5) * db(-19) + pan(shift(octu, 0.013), 0.5) * db(-19))
    # pre-chorus: rising pitched backing (octave up, filtered)
    pre_env = region_env(n, PRE)
    pre_src = lead_in * pre_env
    pre_up = graillon(pre_src, smooth=12.0, corr_amount=60.0, pitch_shift_st=12.0, formant=True)
    pre_up = board(pre_up, pb.HighpassFilter(700), pb.Compressor(threshold_db=-24, ratio=5))
    ramp = np.zeros(n, np.float32)
    for a, b in PRE:
        i, j = int(a * SR), int(b * SR)
        ramp[i:j] = np.linspace(0.25, 1.0, j - i)
        k2 = min(n, j + int(0.4 * SR))
        ramp[j:k2] = np.linspace(1.0, 0.0, k2 - j)      # let tails ring out, no hard cut
    hook += (pan(pre_up, -0.6) + pan(shift(pre_up, 0.021), 0.6)) * ramp * db(-12)

    # ---------- OWN ADLIBS: echo the last word into vocal gaps ----------
    env = envelope(lead_in)
    thr = db(-24) * env.max()
    voiced = env > thr
    ad = np.zeros((2, n), np.float32)
    adlib_times = []
    styles = ["up", "phone", "down", "up", "phone"]
    k = 0
    regions = VERSES + [BRIDGE]
    for a, b in regions:
        i0, i1 = int(a * SR), int(b * SR)
        v = voiced[i0:i1]
        # find gaps: falling edges followed by >= 0.30 s of silence
        edges = np.flatnonzero(v[:-1] & ~v[1:])
        last = -1e9
        for e in edges:
            g_end = np.flatnonzero(v[e + 1:])
            gap = (g_end[0] if len(g_end) else len(v) - e) / SR
            t_end = (i0 + e) / SR
            if gap < 0.30 or t_end - last < bar / 2:
                continue
            src = voc[:, int((t_end - 0.32) * SR):int(t_end * SR)]
            if np.abs(src).max() < thr:
                continue
            st = styles[k % len(styles)]
            w = np.ones(src.shape[1], np.float32)
            nf = int(0.12 * SR)
            w[-nf:] = np.linspace(1, 0, nf)
            w[:int(0.005 * SR)] = np.linspace(0, 1, int(0.005 * SR))
            src = (src * w).astype(np.float32)
            src = np.pad(src, ((0, 0), (0, int(1.2 * SR))))
            if st == "up":
                x = graillon(src, smooth=8.0, corr_amount=60.0, pitch_shift_st=12.0, formant=True)
                x = board(x, pb.HighpassFilter(400), pb.Delay(delay_seconds=bar / 8 * 3 / 2, feedback=0.35, mix=0.45))
                x = pan(x, -0.75 if k % 2 else 0.75) * db(-9)
            elif st == "down":
                x = graillon(src, smooth=8.0, corr_amount=60.0, pitch_shift_st=-12.0)
                x = board(x, pb.Distortion(drive_db=8), pb.LowpassFilter(4000), pb.Delay(delay_seconds=bar / 4, feedback=0.25, mix=0.35))
                x = pan(x, 0.0) * db(-10)
            else:
                x = radio(graillon(src, smooth=8.0, corr_amount=60.0))
                x = board(x, pb.Delay(delay_seconds=bar / 8, feedback=0.45, mix=0.5))
                x = pan(x, 0.6 if k % 2 else -0.6) * db(-8)
            pos = int((t_end + 0.06) * SR)
            m = min(x.shape[1], n - pos)
            ad[:, pos:pos + m] += x[:, :m]
            adlib_times.append((round(t_end, 2), st))
            last = t_end
            k += 1
    log["adlibs"] = adlib_times

    # ---------- HOOK PART: vocal chop melody ----------
    # take a sustained syllable from chorus 2 ("Galaxie"), retune to C# minor arpeggio
    seg = lead_t[:, int(67.2 * SR):int(68.9 * SR)].mean(0)
    f0, vflag, _ = librosa.pyin(seg, fmin=80, fmax=600, sr=SR, frame_length=2048, hop_length=256)
    fr = np.flatnonzero(vflag)
    # longest voiced run
    runs = np.split(fr, np.flatnonzero(np.diff(fr) > 1) + 1)
    run = max(runs, key=len)
    a_s, b_s = run[0] * 256, min(len(seg), run[-1] * 256 + 2048)
    syl = seg[a_s:b_s]
    syl_f0 = float(np.nanmedian(f0[run]))
    syl_midi = librosa.hz_to_midi(syl_f0)
    log["chop_source"] = {"t": round(67.2 + a_s / SR, 3), "len": round(len(syl) / SR, 3), "midi": round(float(syl_midi), 2)}
    pattern = [61, None, 64, 61, 68, None, 66, 64,  61, None, 64, 68, 71, 68, 66, None]   # C#4 E4 G#4 B4 (one octave above source)
    eighth = bar / 8
    chop_len = int(min(len(syl), eighth * 0.92 * SR))
    fi, fo_ = int(0.005 * SR), int(0.03 * SR)
    env_c = np.concatenate([np.linspace(0, 1, fi), np.ones(chop_len - fi - fo_), np.linspace(1, 0, fo_)])
    base_chop = syl[:chop_len] * env_c[:chop_len]
    cache = {}

    def chop(note):
        if note not in cache:
            cache[note] = librosa.effects.pitch_shift(base_chop, sr=SR, n_steps=float(note - syl_midi)).astype(np.float32)
        return cache[note]

    chops = np.zeros((2, n), np.float32)
    chop_regions = [(HOOKS[2][0], HOOKS[2][1]), (124.0, 136.0), (OUTRO[0] + 4.0, OUTRO[1] - 2.0)]
    for a, b in chop_regions:
        # eighth grid interpolated between measured beats (tempo drifts 96.6-97.1 BPM);
        # beats[0::4] are downbeats (kick on phase 0, snare on 1/3, see downbeat.py)
        bi = int(np.searchsorted(beats, a))
        while bi % 4 != 0 and bi < len(beats):
            bi += 1
        step = 0
        while bi + step // 2 + 1 < len(beats):
            j = bi + step // 2
            t = beats[j] if step % 2 == 0 else 0.5 * (beats[j] + beats[j + 1])
            if t >= b - eighth:
                break
            note = pattern[step % 16]
            if note is not None:
                c = chop(note)
                pos = int(t * SR)
                m = min(len(c), n - pos)
                p_ = -0.45 if step % 2 else 0.45
                chops[:, pos:pos + m] += pan(c[:m], p_)
            step += 1
    chops = board(chops, pb.HighpassFilter(250), pb.LowpassFilter(9000),
                  pb.Delay(delay_seconds=eighth * 3, feedback=0.3, mix=0.25),
                  pb.Reverb(room_size=0.8, wet_level=0.35, dry_level=0.75, width=1.0))
    chops *= db(-13)

    # ---------- RADIO INTRO: producer tag from v1 ----------
    tag = load(S + "/v1/vocals.wav")[:, : int(3.2 * SR)]
    tag = radio(tag)
    tag_full = np.zeros((2, n), np.float32)
    pos = int(1.0 * SR)
    tag_full[:, pos:pos + tag.shape[1]] = tag * db(-4)
    # static bed under the intro, fading out into the first chorus line
    stat_n = int(9.0 * SR)
    sb = board(np.stack([noise(stat_n, 11), noise(stat_n, 12)]) * 0.01, pb.HighpassFilter(1500), pb.LowpassFilter(7000))
    sb *= np.concatenate([np.linspace(0, 1, int(0.5 * SR)), np.ones(stat_n - int(4.5 * SR)), np.linspace(1, 0, int(4.0 * SR))])
    tag_full[:, :stat_n] += sb

    # ---------- FX: risers + impacts ----------
    sfx = np.zeros((2, n), np.float32)
    for d in drops:
        r = riser(2 * bar, seed=int(d))
        i = int(d * SR) - r.shape[1]
        if i > 0:
            sfx[:, i:i + r.shape[1]] += r * db(-21)
        im = impact(seed=int(d) + 7)
        j = int(d * SR)
        m = min(im.shape[1], n - j)
        sfx[:, j:j + m] += im[:, :m] * db(-22)
    log["drops"] = drops

    # ---------- BEAT ----------
    drm_p = board(drm, pb.HighpassFilter(28), pb.PeakFilter(60, 1.5, 1.0), pb.PeakFilter(420, -2, 1.2),
                  pb.HighShelfFilter(8000, 1.5))
    par = board(drm_p, pb.Compressor(threshold_db=-28, ratio=8, attack_ms=8, release_ms=90), pb.Distortion(drive_db=6))
    drums = drm_p + par * db(-8)
    drums = softclip(drums * db(3.0), -5.0)        # trap-style drum clipper, ~3 dB off the hits
    # 808: harmonics for small speakers, mono low end
    bass_s = board(bas.mean(0, keepdims=True).repeat(2, 0), pb.HighpassFilter(25), pb.LowShelfFilter(55, 3.5))
    harm = board(bass_s, pb.Distortion(drive_db=12), pb.HighpassFilter(140), pb.LowpassFilter(1800))
    bass_out = board(bass_s + harm * db(-9), pb.Compressor(threshold_db=-18, ratio=4, attack_ms=10, release_ms=150))
    bass_out = softclip(bass_out * db(2.0), -7.0)
    # sidechain-ish duck of music bed by kick
    b, a = ss.butter(4, 110 / (SR / 2))
    kick = np.abs(ss.filtfilt(b, a, drm.mean(0)))
    ke = ss.lfilter([1 - 0.9995], [1, -0.9995], kick)
    ke = ke / (np.percentile(ke, 99.5) + 1e-9)
    duck = 1 - 0.18 * np.clip(ke, 0, 1)
    oth_p = board(oth, pb.HighpassFilter(110), pb.PeakFilter(2800, -1.5, 1.0), pb.HighShelfFilter(11000, 1.5))
    M, Sd = oth_p.mean(0), (oth_p[0] - oth_p[1]) / 2
    Sd *= 1.5
    oth_w = np.stack([M + Sd, M - Sd]) * duck

    # ---------- VOCAL FX BUS ----------
    vox_dry = lead + hook
    plate = board(vox_dry, pb.HighpassFilter(300), pb.LowpassFilter(9000),
                  pb.Reverb(room_size=0.62, damping=0.5, wet_level=1.0, dry_level=0.0, width=1.0))
    dly = board(lead * hook_env, pb.HighpassFilter(500), pb.Delay(delay_seconds=bar / 4 * 0.75, feedback=0.3, mix=1.0), pb.LowpassFilter(6000))
    vox = vox_dry + plate * db(-16) + dly * db(-20) + ad + chops

    # bridge: drums out already in source; final chorus drum-cut handled by source arrangement
    mix = drums * db(-1.0) + bass_out * db(-0.5) + oth_w * db(2.0) + vox * db(0.5) + sfx + tag_full
    # gentle ducking of the bed under lead vocal presence (keeps words on top)
    ve = envelope(lead, win=0.05)
    ve = ve / (np.percentile(ve, 99) + 1e-9)
    mix -= oth_w * db(2.0) * (0.12 * np.clip(ve, 0, 1))   # bed dips ~1 dB under the lead

    # ---------- MASTER ----------
    m = board(mix, pb.HighpassFilter(22),
              pb.Compressor(threshold_db=-18, ratio=2.0, attack_ms=30, release_ms=200),
              pb.LowShelfFilter(80, 1.2))
    # mono below 100 Hz
    b, a = ss.butter(4, 100 / (SR / 2))
    low = ss.filtfilt(b, a, m, axis=1)
    m = (m - low) + low.mean(0, keepdims=True)
    # fade out tail
    end = int((OUTRO[1] + 1.2) * SR)
    m = m[:, :end]
    fo = int(3.0 * SR)
    m[:, -fo:] *= np.linspace(1, 0, fo) ** 1.5
    m[:, :int(0.01 * SR)] *= np.linspace(0, 1, int(0.01 * SR))

    # loudness: iterate gain into limiter until integrated loudness hits target
    gain = args.lufs - lufs(m)
    for _ in range(8):
        used = gain
        raw = m * db(used)
        pre = softclip(raw, -0.6)              # bus clipper shaves the last transients before the limiter
        out = true_peak_limit(pre, ceiling_db=-1.2)
        L = lufs(out)
        if abs(L - args.lufs) < 0.15:
            break
        gain += args.lufs - L

    def red(a, b):
        return 20 * np.log10(np.maximum(np.abs(a).max(0), 1e-6) / np.maximum(np.abs(b).max(0), 1e-6))
    loud = np.abs(raw).max(0) > db(-12)
    clip_gr, lim_gr = red(raw, pre)[loud], red(pre, out)[loud]
    log["master"] = {"lufs": round(float(L), 2), "gain_db": round(float(used), 2),
                     "clipper_gr_db_median/p99": [round(float(np.median(clip_gr)), 2), round(float(np.percentile(clip_gr, 99)), 2)],
                     "limiter_gr_db_median/p99": [round(float(np.median(lim_gr)), 2), round(float(np.percentile(lim_gr, 99)), 2)]}

    R = P + "/renders"
    sf.write(R + "/" + args.name, out.T, SR, subtype="PCM_24")
    stems = {"lead_vocal": lead, "hook_stack": hook, "adlibs": ad, "vocal_chops": chops,
             "vocal_fx": plate * db(-16) + dly * db(-20),
             "drums": drums, "bass": bass_out, "music": oth_w, "fx": sfx + tag_full}   # pre bus-gain / master
    for k_, v_ in stems.items():
        sf.write(f"{R}/stems/{k_}.wav", fit(v_, end).T, SR, subtype="PCM_24")
    with open(P + "/analyse/produce_log.json", "w") as fh:
        json.dump(log, fh, indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
