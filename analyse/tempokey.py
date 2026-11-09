import sys, numpy as np, librosa
NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
maj=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]); mi=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
for f in sys.argv[1:]:
    y,sr=librosa.load(f,sr=22050,mono=True)
    oenv=librosa.onset.onset_strength(y=y,sr=sr,hop_length=64)
    t,beats=librosa.beat.beat_track(onset_envelope=oenv,sr=sr,hop_length=64)
    bt=librosa.frames_to_time(beats,sr=sr,hop_length=64)
    a,b=np.polyfit(np.arange(len(bt)),bt,1)
    c=librosa.feature.chroma_cqt(y=y,sr=sr).mean(1)
    sc=[(np.corrcoef(np.roll(maj,k),c)[0,1],NAMES[k]+' Dur') for k in range(12)]+[(np.corrcoef(np.roll(mi,k),c)[0,1],NAMES[k]+' Moll') for k in range(12)]
    sc.sort(reverse=True)
    print(f, 'BPM~%.2f'%(60/a), 'erster Beat %.3f'%b, sc[:3])
