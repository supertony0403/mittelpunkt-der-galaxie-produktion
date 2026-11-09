import sys, numpy as np, librosa, json
f=sys.argv[1]
y,sr=librosa.load(f,sr=22050,mono=True)
hop=64
oenv=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
for st in (99,98,49.5):
    t,beats=librosa.beat.beat_track(onset_envelope=oenv,sr=sr,hop_length=hop,start_bpm=st,tightness=400)
    bt=librosa.frames_to_time(beats,sr=sr,hop_length=hop)
    a,b=np.polyfit(np.arange(len(bt)),bt,1); res=bt-(a*np.arange(len(bt))+b)
    print('start',st,'BPM %.3f'%(60/a),'phase %.4f'%(b%a),'resid ms %.1f'%(1000*np.std(res)), 'n',len(bt))
# RMS envelope per 2s to see sections
rms=librosa.feature.rms(y=y,hop_length=2048)[0]; tt=librosa.frames_to_time(np.arange(len(rms)),sr=sr,hop_length=2048)
for s in range(0,int(tt[-1]),4):
    m=rms[(tt>=s)&(tt<s+4)].mean(); print(f'{s:4d}s {20*np.log10(m+1e-9):6.1f} dB', '#'*int(max(0,40+20*np.log10(m+1e-9))))
