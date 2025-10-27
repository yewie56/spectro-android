# GitHub Actions: Build Kivy/Python Android App (Buildozer)

Diese Workflows bauen eure **Kivy/Python**-App mit **Buildozer** auf **Ubuntu-Runnern** in GitHub Actions.

## Dateien
- `.github/workflows/android-debug.yml` – Debug-APK auf `push` (main/master) und via *Run workflow*.
- `.github/workflows/android-release.yml` – Release-APK oder -AAB via *Run workflow*. Optionales Signieren via Secrets.

## Voraussetzungen im Repository
- Projekt-Root enthält `buildozer.spec` und euren App-Code (`main.py`, etc.).
- In `buildozer.spec` sind Android-APIs und Requirements korrekt gesetzt (z. B. `python3,kivy,numpy,android,pyjnius`).

## Nutzung (Debug)
1. Commit & Push nach `main`/`master` oder **Actions → Android APK (Buildozer Debug) → Run workflow**.
2. Artefakt **apk-debug** herunterladen (liegt im `bin/`-Ordner).

## Nutzung (Release, optional signiert)
1. Hinterlegt folgende **Repository Secrets** (Settings → Secrets and variables → Actions → New repository secret):
   - `ANDROID_KEYSTORE_BASE64` – euer Keystore als **Base64**-String (siehe unten).
   - `ANDROID_KEYSTORE_PASSWORD` – Passwort des Keystores.
   - `ANDROID_KEY_ALIAS` – Alias-Name des Schlüssels.
   - `ANDROID_KEY_ALIAS_PASSWORD` – Passwort des Alias.
2. (Optional) Für AAB in `buildozer.spec` setzen:
   ```ini
   [app]
   android.bundle = True
   ```
3. In **Actions → Android APK/AAB (Buildozer Release)** auf **Run workflow** klicken und optional `build_bundle: true` wählen.
4. Artefakte (`.apk`/`.aab`) unter **release-…** herunterladen.

### Keystore als Base64 erzeugen
Auf eurem Rechner:
```bash
# vorhandene keystore.jks in Base64 umwandeln
base64 -w 0 keystore.jks > keystore.jks.b64
# Inhalt der Datei keystore.jks.b64 als Secret ANDROID_KEYSTORE_BASE64 einfügen
```
> Unter macOS: `base64 keystore.jks > keystore.jks.b64`

## Caching
Beide Workflows cachen `.buildozer/` und `~/.gradle/`, um Builds zu beschleunigen.

## Typische Fehler
- **SDK/NDK Download schlägt fehl** → erneut ausführen (temporäre Netzwerkprobleme).
- **Java-Version**: Wir installieren `openjdk-17-jdk`. Sollte mit aktuellen Toolchains kompatibel sein.
- **Requirements**: Stimmt die `requirements`-Zeile in `buildozer.spec`? Fehlende Libs führen zu Build-Fehlern.

## Anpassungen
- Python-Version in `actions/setup-python` (default: 3.11).
- Android-API/NDK in `buildozer.spec`.
- Falls ihr zusätzliche Systempakete braucht, ergänzt die `apt-get install`-Zeile.
