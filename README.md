# Mittelpunkt der Galaxie – Produktionsskripte

Skriptgesteuerte Nachproduktion eines generierten Songs. Aus den Demucs-Stems entstehen Mix und Master.
Audio, Stems, Text und Cover sind bewusst **nicht** im Repo, weil der Song unveröffentlicht ist.

| Datei | Zweck |
|---|---|
| `analyse/produce.py` | kompletter Mix und Master: Graillon-Autotune (cis-Moll), Hook-Doubles und Oktaven, eigene Echo-Adlibs in Gesangslücken, Vocal-Chop-Hook, Radio-Intro mit Producer-Tag, Riser und Impacts, Drum- und 808-Clipper, True-Peak-Limiter mit Lautheitsregelung |
| `analyse/x42tune.py` | minimaler ctypes-LV2-Host für x42-Autotune (`fat1`), ohne DAW |
| `analyse/qc.py` | Messung: Clipper-Arbeit, Tonhöhe, Spektrum gegen das Original, Stereo/Mono |
| `analyse/grid2.py`, `downbeat.py` | Beat-Raster und Taktanfänge aus der Drum-Spur |
| `analyse/transcribe.py` | faster-whisper-Transkript zur Abschnittsbestimmung |

Ablauf:

```sh
~/opt/stemsplit/.venv/bin/demucs -n htdemucs_ft --int24 -o stems src/v3.mp3 src/v1.mp3
~/opt/stemsplit/.venv/bin/python -P analyse/grid2.py stems/htdemucs_ft/v3/drums.wav analyse/v3_beats.npy
~/opt/stemsplit/.venv/bin/python -P analyse/produce.py "$PWD" --lufs -10
~/opt/stemsplit/.venv/bin/python -P analyse/qc.py "$PWD"
```
