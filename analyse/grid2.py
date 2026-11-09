import sys, numpy as np, librosa
f=sys.argv[1]
y,sr=librosa.load(f,sr=22050,mono=True)
hop=64
oenv=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
t,beats=librosa.beat.beat_track(onset_envelope=oenv,sr=sr,hop_length=hop,start_bpm=97,tightness=800)
bt=librosa.frames_to_time(beats,sr=sr,hop_length=hop)
a,b=np.polyfit(np.arange(len(bt)),bt,1); res=bt-(a*np.arange(len(bt))+b)
print('BPM %.3f first %.4f resid %.1f ms n %d'%(60/a,b,1000*np.std(res),len(bt)))
# local tempo over windows of 16 beats
d=np.diff(bt)
for i in range(0,len(d),16):
    w=d[i:i+16]; print('%6.1fs  %.2f BPM  res %+.0f ms'%(bt[i],60/np.median(w), 1000*res[i]))
np.save(sys.argv[2],bt)
