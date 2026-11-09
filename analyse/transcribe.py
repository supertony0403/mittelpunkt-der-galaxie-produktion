import sys
from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8")
for f in sys.argv[1:]:
    print("=====", f, flush=True)
    seg, info = m.transcribe(f, language="de", beam_size=5, vad_filter=True, condition_on_previous_text=False, word_timestamps=False)
    for s in seg:
        print(f"{s.start:6.1f}-{s.end:6.1f} {s.text.strip()}", flush=True)
