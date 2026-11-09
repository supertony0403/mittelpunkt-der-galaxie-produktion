"""Minimal ctypes LV2 host for x42 fat1 (autotune) - offline, mono, latency compensated."""
import ctypes as C
import numpy as np

LIB = "/usr/lib/lv2/fat1.lv2/fat1.so"
BUNDLE = b"/usr/lib/lv2/fat1.lv2/"
URI = b"http://gareus.org/oss/lv2/fat1"


class Feature(C.Structure):
    _fields_ = [("URI", C.c_char_p), ("data", C.c_void_p)]


MAPFN = C.CFUNCTYPE(C.c_uint32, C.c_void_p, C.c_char_p)


class URIDMap(C.Structure):
    _fields_ = [("handle", C.c_void_p), ("map", MAPFN)]


class Descriptor(C.Structure):
    pass


Descriptor._fields_ = [
    ("URI", C.c_char_p),
    ("instantiate", C.CFUNCTYPE(C.c_void_p, C.POINTER(Descriptor), C.c_double, C.c_char_p, C.POINTER(C.POINTER(Feature)))),
    ("connect_port", C.CFUNCTYPE(None, C.c_void_p, C.c_uint32, C.c_void_p)),
    ("activate", C.CFUNCTYPE(None, C.c_void_p)),
    ("run", C.CFUNCTYPE(None, C.c_void_p, C.c_uint32)),
    ("deactivate", C.CFUNCTYPE(None, C.c_void_p)),
    ("cleanup", C.CFUNCTYPE(None, C.c_void_p)),
    ("extension_data", C.c_void_p),
]

_uris = {}


@MAPFN
def _map(handle, uri):
    u = uri.decode()
    if u not in _uris:
        _uris[u] = len(_uris) + 1
    return _uris[u]


# note mask index 0 = C ... 11 = B
CSHARP_MINOR = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]


def tune(x, sr, mask=CSHARP_MINOR, filt=0.02, corr=1.0, fast=1, bias=0.5, offset=0.0, block=1024):
    """x: mono float array. returns (tuned mono array of same length, latency in samples)."""
    lib = C.CDLL(LIB)
    lib.lv2_descriptor.restype = C.POINTER(Descriptor)
    lib.lv2_descriptor.argtypes = [C.c_uint32]
    d = None
    i = 0
    while True:
        p = lib.lv2_descriptor(i)
        if not p:
            break
        if p.contents.URI == URI:
            d = p
            break
        i += 1
    assert d, "fat1 descriptor not found"
    umap = URIDMap(None, _map)
    feat = Feature(b"http://lv2plug.in/ns/ext/urid#map", C.cast(C.pointer(umap), C.c_void_p))
    feats = (C.POINTER(Feature) * 2)(C.pointer(feat), None)
    h = d.contents.instantiate(d, float(sr), BUNDLE, feats)
    assert h, "instantiate failed"

    ctl = {}

    def port(idx, val):
        v = C.c_float(val)
        ctl[idx] = v
        d.contents.connect_port(h, idx, C.byref(v))

    port(3, 0)          # mode auto
    port(4, 0)          # midi channel filter
    port(5, 440.0)      # tuning
    port(6, bias)
    port(7, filt)
    port(8, corr)
    port(9, offset)
    port(10, 2.0)
    port(11, fast)
    for k in range(12):
        port(12 + k, float(mask[k]))
    for k in range(24, 29):
        port(k, 0.0)    # outputs + any extra (scale) ports, so nothing dangles
    # atom sequence input (empty)
    seq = (C.c_uint32 * 64)()
    seq_type = _map(None, b"http://lv2plug.in/ns/ext/atom#Sequence")
    d.contents.connect_port(h, 0, seq)
    inb = (C.c_float * block)()
    outb = (C.c_float * block)()
    d.contents.connect_port(h, 1, inb)
    d.contents.connect_port(h, 2, outb)
    if C.cast(d.contents.activate, C.c_void_p).value:
        d.contents.activate(h)

    x = np.asarray(x, np.float32)
    n = len(x)
    pad = 16384
    xin = np.concatenate([x, np.zeros(pad, np.float32)])
    out = np.zeros_like(xin)
    inp = np.ctypeslib.as_array(inb)
    op = np.ctypeslib.as_array(outb)
    lat = 0
    for s in range(0, len(xin), block):
        m = min(block, len(xin) - s)
        seq[0] = 8          # atom.size (body only)
        seq[1] = seq_type
        seq[2] = 0
        seq[3] = 0
        inp[:m] = xin[s:s + m]
        d.contents.run(h, m)
        out[s:s + m] = op[:m]
        lat = int(ctl[28].value)
    if C.cast(d.contents.deactivate, C.c_void_p).value:
        d.contents.deactivate(h)
    if C.cast(d.contents.cleanup, C.c_void_p).value:
        d.contents.cleanup(h)
    return out[lat:lat + n], lat
