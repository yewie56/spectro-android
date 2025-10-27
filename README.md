# Spectro Waterfall (Android, Kivy)

Ein konfigurierbarer Audio-Spektrumanalyzer mit Wasserfalldiagramm — optimiert für Android (Kivy + Pyjnius), Desktop-Tests via `sounddevice` möglich.

## Features (alles als Optionen)
- Colormap: *grayscale / viridis / inferno*
- Spektrum-Achsen (dB/Frequenz) an/aus
- Peak-Hold (Decay einstellbar)
- Exponentielle Mittelung (Averaging)
- Modi: *FFT / Octave / Third-Octave*
- Sample-Rate, FFT-Size, Hop-Size, dB-Floor/Ceil, Waterfall-Zeilen
- Pause/Resume
- Peak-Frequenz-Anzeige
- SPL-Kalibrier-Offset (dB) additiv
- Optionaler Biquad-Filter: *None / Bandpass / Notch* mit f0 und Q
- Export: CSV (aktuelles Spektrum) und Screenshot (Gesamtfenster)

## Build (Android)
Auf Linux/WSL empfohlen:
```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade buildozer cython
buildozer init  # falls noch nicht vorhanden
# Ersetze buildozer.spec mit der hier gelieferten
buildozer android debug
```
APK liegt unter `bin/`.

## Desktop-Test (optional)
```bash
pip install kivy numpy sounddevice
python main.py
```

## Hinweise
- Manche Geräte bevorzugen `48000 Hz`. Stelle SR entsprechend ein.
- FFT größer ⇒ bessere Frequenzauflösung, höhere Latenz. Hop kleiner ⇒ schnellere Updates, höhere CPU.
- CSV wird im App-Datenordner gespeichert, Screenshot im aktuellen Arbeitsverzeichnis.
