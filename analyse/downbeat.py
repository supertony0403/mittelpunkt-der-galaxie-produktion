import numpy as np, librosa, soundfile as sf, scipy.signal as ss, pedalboard as pb, sys
P=sys.argv[1]
bt=np.load(P+'/analyse/v3_beats.npy')
y,sr=sf.read(P+'/stems/htdemucs_ft/v3/drums.wav'); y=y.mean(1)
b,a=ss.butter(4,120/(sr/2)); lo=ss.filtfilt(b,a,y)
b,a=ss.butter(4,[1500/(sr/2),6000/(sr/2)],'band'); hi=ss.filtfilt(b,a,y)
def e(x,t): i=int(t*sr); return np.abs(x[i:i+int(0.06*sr)]).mean()
K=np.array([e(lo,t) for t in bt]); S=np.array([e(hi,t) for t in bt])
for ph in range(4): print('phase',ph,'kick %.4f snare %.4f'%(K[ph::4].mean(),S[ph::4].mean()))
# graillon latency test
p=pb.load_plugin('/usr/lib/vst3/Auburn Sounds Graillon 3.vst3')
imp=np.zeros((2,sr),np.float32); imp[:,1000:1400]=np.sin(2*np.pi*220*np.arange(400)/sr)
p.correction=False
out=p(imp,sr)
c=ss.correlate(out[0],imp[0],'full'); print('graillon lag samples',np.argmax(c)-(len(imp[0])-1), 'plugin.latency', getattr(p,'reported_latency_samples',None))
