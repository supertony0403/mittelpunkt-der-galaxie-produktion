import numpy as np, soundfile as sf, librosa, scipy.signal as ss, sys
P=sys.argv[1]; SR=44100
m,_=sf.read(P+'/renders/Mittelpunkt der Galaxie - Master 24bit.wav'); m=m.T
o=librosa.load(P+'/src/v3.mp3',sr=SR,mono=False)[0]
if o.ndim==1: o=np.stack([o,o])
# 1) clipper drive: how far would peaks exceed -0.6 dBFS before tanh (invert tanh on output proxy)
import json; print('clipper/limiter GR (from produce_log):', json.load(open(P+'/analyse/produce_log.json'))['master'])
# 2) pitch on tuned lead stem vs C# minor
v,_=sf.read(P+'/renders/stems/lead_vocal.wav'); v=v.mean(1)
f0,vf,_=librosa.pyin(v[int(66*SR):int(118*SR)],fmin=70,fmax=700,sr=SR,frame_length=2048,hop_length=512)
mid=librosa.hz_to_midi(f0[vf]); pcs={1,3,4,6,8,9,11}
near=np.round(mid); dev=np.abs(mid-near)*100; insc=np.isin(near%12,list(pcs))
print('lead pitch: median dev %.0f ct, <30ct %.0f%%, on-scale %.0f%%'%(np.median(dev),100*np.mean(dev<30),100*np.mean(insc)))
vo,_=sf.read(P+'/stems/htdemucs_ft/v3/vocals.wav'); vo=vo.mean(1)
f0,vf,_=librosa.pyin(vo[int(66*SR):int(118*SR)],fmin=70,fmax=700,sr=SR,frame_length=2048,hop_length=512)
mid=librosa.hz_to_midi(f0[vf]); dev=np.abs(mid-np.round(mid))*100
print('raw  pitch: median dev %.0f ct, <30ct %.0f%%'%(np.median(dev),100*np.mean(dev<30)))
# 3) spectrum vs original (level-matched)
def bands(x):
    f,p=ss.welch(x.mean(0),SR,nperseg=8192); out=[]
    for lo,hi,n in [(20,60,'sub'),(60,250,'low'),(250,2000,'mid'),(2000,6000,'presence'),(6000,16000,'air')]:
        out.append((n,10*np.log10(p[(f>=lo)&(f<hi)].mean())))
    return out
import pyloudnorm as pl
mt=pl.Meter(SR); g=mt.integrated_loudness(m.T)-mt.integrated_loudness(o.T)
bm,bo=bands(m),bands(o*10**(g/20))
print('spectrum vs original (level matched):',', '.join('%s %+.1f dB'%(a[0],a[1]-b[1]) for a,b in zip(bm,bo)))
# 4) stereo/mono
print('stereo correlation %.2f'%np.corrcoef(m[0],m[1])[0,1])
mono=m.mean(0); print('mono loss %.2f LU'%(mt.integrated_loudness(m.T)-mt.integrated_loudness(np.stack([mono,mono]).T)))
print('length %.2f s'%(m.shape[1]/SR))
