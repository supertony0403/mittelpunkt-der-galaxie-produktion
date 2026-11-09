# Release-Datenblatt: Mittelpunkt der Galaxie

## Dateien in diesem Ordner

| Datei | Zweck | Prüfwert |
|---|---|---|
| `Mittelpunkt der Galaxie - Master 24bit.wav` | Upload beim Vertrieb (DistroKid, RouteNote, recordJet, …) | 44,1 kHz, 24 Bit, Stereo, 3:37, −10,1 LUFS integriert, −1,2 dBTP, LRA 4,5 LU |
| `Cover 3000x3000.jpg` | Artwork | 3000×3000 px, RGB-JPEG, ohne Text und Logos |
| `Songtext.txt` | Lyrics für Spotify/Musixmatch | an den fertigen Track angepasst (Intro nur mit Producer-Tag) |

Spotify normalisiert auf −14 LUFS. Der Master wird dort also um etwa 4 dB leiser abgespielt und klingt dabei sauber, weil der Limiter kaum eingreift. Bei 0 dBTP würden verlustbehaftete Codecs übersteuern, deshalb liegt die Spitze bei −1,2 dBTP.

## Angaben für das Upload-Formular

| Feld | Wert |
|---|---|
| Titel | Mittelpunkt der Galaxie *(Dateitag bisher: „Galaxie/KK“, bitte festlegen)* |
| Künstler | siso aka. sicksound *(Künstlername genau so wie auf dem Spotify-Profil)* |
| Album/Release | Single, oder Teil von „zweinullzweisechs“ |
| Genre | Hip-Hop/Rap, Untergenre Cloud Rap / Trap |
| Sprache | Deutsch (mit englischen Teilen) |
| **Explicit** | **Ja**: „Fick“, „bitch“, „scheiße“ im Text |
| Erscheinungsjahr | 2026 |
| Producer | SiSo |
| ISRC/UPC | vergibt der Vertrieb kostenlos |
| KI-Angabe | Ja: Gesang und Beat mit Suno generiert, Nachproduktion (Mix/Master, Autotune, Adlibs, Hook-Chops) per Skript |

## ⚠️ Vor dem Upload selbst klären

1. **Suno-Rechte.** Kommerziell veröffentlichen darfst du nur Songs, die du mit einem bezahlten Suno-Abo (Pro oder Premier) erstellt hast, *während das Abo aktiv war*. Mit dem Gratis-Tarif gehören die Rechte Suno, und dann darf der Song nicht auf Spotify. Prüfen in Suno → Song → „Commercial use“.
2. **Cover.** Es stammt ebenfalls aus der Suno-Datei (KI-generiert) und wurde mit Real-ESRGAN (BSD-3-Lizenz) von 360 auf 3000 px hochgerechnet. Es gilt dieselbe Rechtefrage wie bei Punkt 1.
3. **Probehören** auf Kopfhörern *und* Handylautsprecher, bevor du hochlädst. Falls etwas zu laut oder zu leise ist, lassen sich die Pegel aller Ebenen in `analyse/produce.py` ändern; ein neuer Render dauert etwa 1 Minute.
4. **Vertrieb.** Spotify nimmt keine Direkt-Uploads von Einzelkünstlern an, der Weg führt über einen Vertrieb. Plane etwa 2–3 Wochen Vorlauf ein, damit der Pitch an die Spotify-Redaktion („Spotify for Artists“) rechtzeitig geht.
