[app]
title = Spectro Waterfall
package.name = spectro_opts
package.domain = org.example
source.include_exts = py,kv,txt,md
version = 1.1.0

requirements = python3,kivy,numpy,android,pyjnius
orientation = landscape
fullscreen = 1

android.permissions = RECORD_AUDIO, MODIFY_AUDIO_SETTINGS, WAKE_LOCK

android.api = 34
android.minapi = 24
android.ndk_api = 24
android.archs = arm64-v8a, armeabi-v7a

source.dir = .

[buildozer]
log_level = 2

# akzeptiert SDK-Lizenzen non-interaktiv
android.accept_sdk_license = True

# pinne auf eine stabile SDK/API + Build-Tools (vermeidet 36.1)
android.api = 34
android.build_tools_version = 34.0.0
# optional, aber sinnvoll:
# android.minapi = 23

