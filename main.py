# main.py
# Spectrum + Waterfall Analyzer for Android (Kivy + Pyjnius), Desktop fallback via sounddevice.
# Features (all switchable in UI):
# - Colormap: grayscale / viridis / inferno
# - Axes on/off (dB and frequency ticks)
# - Peak-hold with adjustable decay
# - Exponential averaging (magnitude domain)
# - Modes: FFT / Octave / Third-Octave
# - Sample rate, FFT size, Hop size, dB floor/ceil, WF rows
# - Pause/Resume
# - Peak marker (dominant frequency readout)
# - SPL calibration offset (adds dB)
# - Optional biquad pre-filter (None / Bandpass / Notch) with center frequency & Q
# - Export: Save CSV (current spectrum) & Screenshot (entire app window)
#
# Build on Android with buildozer (see README & buildozer.spec).

import threading, time, math, os
import numpy as np

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.utils import platform

# ---------- Defaults ----------
DEFAULT_SR = 48000
DEFAULT_FFT = 2048
DEFAULT_HOP = 1024
DEFAULT_WF_ROWS = 300
DEFAULT_DB_FLOOR = -90.0
DEFAULT_DB_CEIL = 0.0
DEFAULT_SMOOTH_ALPHA = 0.6
DEFAULT_PEAK_DECAY_DB_PER_S = 3.0
DEFAULT_BAND_MODE = 'FFT'  # 'FFT' | 'Octave' | 'Third-Octave'

IS_ANDROID = (platform == 'android')
HAVE_SD = False
if not IS_ANDROID:
    try:
        import sounddevice as sd
        HAVE_SD = True
    except Exception:
        HAVE_SD = False

# ---------- Utility ----------
def db_scale(mag, floor_db, ceil_db, spl_offset_db=0.0):
    mag = np.maximum(mag, 1e-12)
    db = 20.0 * np.log10(mag) + float(spl_offset_db)
    return np.clip(db, floor_db, ceil_db)

def make_colormap_lut(name='grayscale'):
    name = (name or 'grayscale').lower()
    x = np.linspace(0, 1, 256)
    if name == 'grayscale':
        lut = (np.stack([x, x, x], axis=1) * 255).astype(np.uint8)
    elif name == 'viridis':
        r = np.clip(0.2777 + 2.185*x - 2.162*x*x, 0, 1)
        g = np.clip(0.005 - 0.196*x + 1.861*x*x - 1.402*x**3, 0, 1)
        b = np.clip(0.497 + 1.026*x - 2.344*x*x + 1.834*x**3 - 0.514*x**4, 0, 1)
        lut = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)
    elif name == 'inferno':
        r = np.clip(np.power(x, 0.7), 0, 1)
        g = np.clip(np.power(x, 1.5), 0, 1)
        b = np.clip(np.power(x, 3.0), 0, 1)
        lut = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)
    else:
        lut = (np.stack([x, x, x], axis=1) * 255).astype(np.uint8)
    return lut

def bins_for_bands(fmin=20.0, fmax=20000.0, sr=DEFAULT_SR, nfft=DEFAULT_FFT, mode='Octave'):
    if mode not in ('Octave', 'Third-Octave'):
        return None
    k = 1 if mode == 'Octave' else 3
    bands = []
    fc = 1000.0
    while fc > fmin:
        fc /= 2**(1.0/k)
    while fc < fmax:
        fl = fc / (2**(1/(2*k)))
        fu = fc * (2**(1/(2*k)))
        bin_l = int(np.floor(fl * nfft / sr))
        bin_u = int(np.ceil(fu * nfft / sr))
        bin_l = max(1, bin_l)
        bin_u = min(nfft//2, bin_u)
        if bin_u > bin_l:
            bands.append((bin_l, bin_u, fc))
        fc *= 2**(1.0/k)
    return bands

# Simple biquad filter based on RBJ cookbook (direct form I)
class Biquad:
    def __init__(self):
        self.b0=self.b1=self.b2=1.0
        self.a0=self.a1=self.a2=0.0
        self.z1=self.z2=0.0

    def set_notch(self, sr, f0, Q):
        omega = 2*math.pi*f0/sr
        alpha = math.sin(omega)/(2*Q)
        b0=1; b1=-2*math.cos(omega); b2=1
        a0=1+alpha; a1=-2*math.cos(omega); a2=1-alpha
        self._set(b0,b1,b2,a0,a1,a2)

    def set_bandpass(self, sr, f0, Q):
        omega = 2*math.pi*f0/sr
        alpha = math.sin(omega)/(2*Q)
        b0=alpha; b1=0; b2=-alpha
        a0=1+alpha; a1=-2*math.cos(omega); a2=1-alpha
        self._set(b0,b1,b2,a0,a1,a2)

    def _set(self,b0,b1,b2,a0,a1,a2):
        self.b0=b0/a0; self.b1=b1/a0; self.b2=b2/a0
        self.a1=a1/a0; self.a2=a2/a0
        self.z1=self.z2=0.0

    def process(self, x):
        # x: np.array float32
        y = np.empty_like(x)
        b0,b1,b2,a1,a2 = self.b0,self.b1,self.b2,self.a1,self.a2
        z1,z2 = self.z1,self.z2
        for i,xi in enumerate(x):
            yi = b0*xi + z1
            z1 = b1*xi - a1*yi + z2
            z2 = b2*xi - a2*yi
            y[i]=yi
        self.z1,self.z2 = z1,z2
        return y

# ---------- Widgets ----------
class Waterfall(Widget):
    def __init__(self, n_bins, rows=DEFAULT_WF_ROWS, colormap='grayscale', **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.rows = rows
        self.set_colormap(colormap)
        self.buf = np.zeros((rows, self.n_bins, 3), dtype=np.uint8)
        self.texture = Texture.create(size=(self.n_bins, self.rows), colorfmt='rgb', bufferfmt='ubyte')
        self.texture.mag_filter = 'nearest'; self.texture.min_filter='nearest'
        with self.canvas:
            Color(1,1,1,1)
            self.rect = Rectangle(texture=self.texture, pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)
        self._refresh()

    def _upd(self, *a):
        self.rect.pos = self.pos; self.rect.size=self.size

    def set_colormap(self, name):
        self.lut_name = name
        self.lut = make_colormap_lut(name)

    def push_row_u8(self, row_u8):
        if row_u8.shape[0] != self.n_bins: return
        rgb = self.lut[row_u8]
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1,:,:] = rgb
        self._refresh()

    def _refresh(self):
        self.texture.blit_buffer(self.buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

class Spectrum(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spec_db=None; self.peak_db=None
        self.show_axes=True
        self.db_floor=DEFAULT_DB_FLOOR; self.db_ceil=DEFAULT_DB_CEIL
        self.freq_bins_hz=None
        self.marker_text=""
        self.bind(size=self.redraw, pos=self.redraw)

    def set_freq_axis(self, sr, nfft):
        self.freq_bins_hz = np.linspace(0, sr/2, nfft//2 + 1)

    def set_range(self, floor, ceil):
        self.db_floor=floor; self.db_ceil=ceil

    def set_axes(self, show):
        self.show_axes=show; self.redraw()

    def set_spectrum(self, spec_db, peak_db=None, marker_text=""):
        self.spec_db=spec_db; self.peak_db=peak_db; self.marker_text=marker_text
        self.redraw()

    def redraw(self, *a):
        self.canvas.clear()
        if self.spec_db is None or len(self.spec_db)<2: return
        w,h = self.width, self.height
        n = len(self.spec_db)

        def map_db_to_y(db):
            y = (db - self.db_floor)/(self.db_ceil-self.db_floor+1e-9)
            return np.clip(y,0,1)*(h-6)+3

        xs = np.linspace(0, w, n)

        with self.canvas:
            if self.show_axes:
                Color(0.3,0.3,0.3,1)
                Line(points=[self.x+2,self.y+2,self.x+w-2,self.y+2], width=1)
                Line(points=[self.x+2,self.y+2,self.x+2,self.y+h-2], width=1)
                for d in range(int(self.db_floor), int(self.db_ceil)+1, 10):
                    yy = self.y + map_db_to_y(d) - self.y
                    Line(points=[self.x+2,yy,self.x+w-2,yy], width=0.5)
                if self.freq_bins_hz is not None and len(self.freq_bins_hz)==n:
                    for f in [100,200,500,1000,2000,5000,10000,20000]:
                        idx = int(np.argmin(np.abs(self.freq_bins_hz - f)))
                        xx = self.x + xs[idx]
                        Line(points=[xx, self.y+2, xx, self.y+h-2], width=0.5)

            # spectrum
            Color(0.2,0.8,1.0,1)
            yvals = map_db_to_y(self.spec_db)
            pts = []
            for i in range(n):
                pts.extend([self.x + xs[i], self.y + yvals[i]])
            Line(points=pts, width=1.4)

            # peak-hold
            if self.peak_db is not None:
                Color(1.0,0.6,0.2,1)
                ypeak = map_db_to_y(self.peak_db)
                pts2=[]
                for i in range(n):
                    pts2.extend([self.x + xs[i], self.y + ypeak[i]])
                Line(points=pts2, width=1.0)

            # marker text (top-left)
            if self.marker_text:
                # draw a small underline at peak x too (optional)
                pass  # Keep text minimal to avoid heavy Label in canvas

# ---------- Audio + DSP ----------
class AudioWorker:
    def __init__(self, on_frame_cb,
                 sr=DEFAULT_SR, fft_size=DEFAULT_FFT, hop_size=DEFAULT_HOP,
                 db_floor=DEFAULT_DB_FLOOR, db_ceil=DEFAULT_DB_CEIL,
                 smooth_alpha=DEFAULT_SMOOTH_ALPHA,
                 peak_decay_db_s=DEFAULT_PEAK_DECAY_DB_PER_S,
                 band_mode=DEFAULT_BAND_MODE,
                 spl_offset_db=0.0,
                 filt_type='None', filt_f0=1000.0, filt_Q=3.0):
        self.on_frame_cb = on_frame_cb
        self.sr=sr; self.fft_size=fft_size; self.hop_size=hop_size
        self.db_floor=db_floor; self.db_ceil=db_ceil
        self.smooth_alpha=smooth_alpha
        self.peak_decay_db_s=peak_decay_db_s
        self.band_mode=band_mode
        self.spl_offset_db=spl_offset_db
        self.win = np.hanning(self.fft_size).astype(np.float32)
        self.running=False; self.thread=None; self.paused=False
        self.prev_mag=None; self.prev_band_mag=None; self.peak_db=None
        self.last_t=time.time()
        self.band_map = None
        self.filter = None
        self._set_filter(filt_type, filt_f0, filt_Q)

    def _set_filter(self, ftype, f0, Q):
        ftype = (ftype or 'None')
        if ftype=='None':
            self.filter=None
        else:
            bq = Biquad()
            if ftype=='Notch':
                bq.set_notch(self.sr, max(1.0, f0), max(0.1, Q))
            else:
                bq.set_bandpass(self.sr, max(1.0, f0), max(0.1, Q))
            self.filter=bq

    def restart(self, **kwargs):
        was_running = self.running
        self.stop()
        for k,v in kwargs.items(): setattr(self,k,v)
        self.win = np.hanning(self.fft_size).astype(np.float32)
        self.prev_mag=None; self.prev_band_mag=None; self.peak_db=None
        self.band_map=None
        self._set_filter(kwargs.get('filt_type','None'), kwargs.get('filt_f0',1000.0), kwargs.get('filt_Q',3.0))
        if was_running: self.start()

    def pause(self, flag): self.paused=bool(flag)

    def _emit(self, db_vec, marker_txt):
        norm = (db_vec - self.db_floor)/(self.db_ceil - self.db_floor + 1e-9)
        row_u8 = np.clip(norm*255.0, 0, 255).astype(np.uint8)
        Clock.schedule_once(lambda dt: self.on_frame_cb(db_vec.copy(), row_u8.copy(), self.peak_db.copy() if self.peak_db is not None else None, marker_txt))

    def _android_loop(self):
        from jnius import autoclass
        from android.permissions import request_permissions, Permission, check_permission
        AudioFormat   = autoclass('android.media.AudioFormat')
        AudioRecord   = autoclass('android.media.AudioRecord')
        MediaRecorder = autoclass('android.media.MediaRecorder$AudioSource')
        if not check_permission(Permission.RECORD_AUDIO):
            request_permissions([Permission.RECORD_AUDIO]); time.sleep(1.0)
        channel = AudioFormat.CHANNEL_IN_MONO; encoding=AudioFormat.ENCODING_PCM_16BIT
        min_buf = AudioRecord.getMinBufferSize(self.sr, channel, encoding)
        buff_size = max(min_buf, self.hop_size*4)
        rec = AudioRecord(MediaRecorder.MIC, self.sr, channel, encoding, buff_size)
        if rec.getState()!=AudioRecord.STATE_INITIALIZED:
            print("AudioRecord init failed"); return
        rec.startRecording()
        pcm = np.zeros(self.hop_size, dtype=np.int16); ring = np.zeros(0, dtype=np.int16)
        try:
            while self.running:
                if self.paused: time.sleep(0.02); continue
                read = rec.read(pcm, 0, self.hop_size)
                if read>0:
                    ring = np.concatenate([ring, pcm[:read]])
                    while len(ring)>=self.fft_size:
                        frame = ring[:self.fft_size].astype(np.float32)/32768.0
                        ring = ring[self.hop_size:]
                        self._process_frame(frame)
                else:
                    time.sleep(0.005)
        finally:
            try: rec.stop()
            except Exception: pass
            rec.release()

    def _desktop_loop(self):
        if not HAVE_SD: print("sounddevice not available."); return
        q=[]; ev=threading.Event()
        def cb(indata, frames, time_info, status):
            q.append(indata.copy().flatten()); ev.set()
        stream=None
        try:
            stream = sd.InputStream(channels=1, samplerate=self.sr, blocksize=self.hop_size, dtype='float32', callback=cb)
            stream.start(); ring = np.zeros(0, dtype=np.float32)
            while self.running:
                if self.paused: time.sleep(0.02); continue
                if not q: ev.wait(0.02); ev.clear(); continue
                chunk = q.pop(0); ring = np.concatenate([ring, chunk])
                while len(ring)>=self.fft_size:
                    frame = ring[:self.fft_size]; ring = ring[self.hop_size:]
                    self._process_frame(frame)
        finally:
            if stream is not None: stream.stop(); stream.close()

    def _process_frame(self, frame_f32):
        if self.filter is not None:
            frame_f32 = self.filter.process(frame_f32.astype(np.float32))

        xw = frame_f32 * self.win
        fft = np.fft.rfft(xw.astype(np.float32))
        mag = np.abs(fft)

        now = time.time(); dt = max(1e-3, now - self.last_t); self.last_t = now

        if self.band_mode=='FFT':
            if self.prev_mag is None: self.prev_mag = mag
            else:
                a = np.clip(self.smooth_alpha, 0.0, 0.9999)
                self.prev_mag = a*self.prev_mag + (1.0-a)*mag
            mag_s = self.prev_mag
            db_vec = db_scale(mag_s, self.db_floor, self.db_ceil, self.spl_offset_db)
            # marker: peak bin
            pk_idx = int(np.argmax(db_vec))
            pk_freq = pk_idx * self.sr / self.fft_size
        else:
            if self.band_map is None:
                mm = 'Octave' if self.band_mode=='Octave' else 'Third-Octave'
                self.band_map = bins_for_bands(sr=self.sr, nfft=self.fft_size, mode=mm)
            vals=[]; centers=[]
            for (b0,b1,fc) in self.band_map:
                bm = np.sqrt(np.mean((mag[b0:b1+1])**2))
                vals.append(bm); centers.append(fc)
            vals=np.array(vals, dtype=np.float32); centers=np.array(centers, dtype=np.float32)
            if self.prev_band_mag is None: self.prev_band_mag = vals
            else:
                a = np.clip(self.smooth_alpha, 0.0, 0.9999)
                self.prev_band_mag = a*self.prev_band_mag + (1.0-a)*vals
            db_vec = db_scale(self.prev_band_mag, self.db_floor, self.db_ceil, self.spl_offset_db)
            pk_idx = int(np.argmax(db_vec)); pk_freq = centers[pk_idx] if len(centers)>0 else 0.0

        # peak-hold
        if self.peak_decay_db_s>0:
            if self.peak_db is None or len(self.peak_db)!=len(db_vec):
                self.peak_db = db_vec.copy()
            self.peak_db -= self.peak_decay_db_s * dt
            self.peak_db = np.maximum(self.peak_db, db_vec)
        else:
            self.peak_db=None

        marker_txt = f"Peak: {pk_freq:0.1f} Hz"
        self._emit(db_vec, marker_txt)

    def start(self):
        if self.running: return
        self.running=True
        self.thread = threading.Thread(target=self._android_loop if IS_ANDROID else self._desktop_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running=False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread=None

# ---------- Controls ----------
class ControlPanel(GridLayout):
    def __init__(self, on_change, on_save_csv, on_screenshot, **kwargs):
        super().__init__(cols=6, size_hint=(1,None), height=210, padding=6, spacing=6, **kwargs)
        self.on_change = on_change
        self.on_save_csv = on_save_csv
        self.on_screenshot = on_screenshot

        # Row 1
        self.add_widget(Label(text="Colormap", size_hint=(None,None), size=(100,28)))
        self.sp_cmap = Spinner(text='inferno', values=('grayscale','viridis','inferno'), size_hint=(None,None), size=(120,28)); self.sp_cmap.bind(text=lambda *a: self.on_change())
        self.add_widget(self.sp_cmap)

        self.add_widget(Label(text="Axes", size_hint=(None,None), size=(80,28)))
        self.cb_axes = CheckBox(active=True); self.cb_axes.bind(active=lambda *a: self.on_change()); self.add_widget(self.cb_axes)

        self.add_widget(Label(text="Mode", size_hint=(None,None), size=(80,28)))
        self.sp_mode = Spinner(text='FFT', values=('FFT','Octave','Third-Octave'), size_hint=(None,None), size=(130,28)); self.sp_mode.bind(text=lambda *a: self.on_change())
        self.add_widget(self.sp_mode)

        self.add_widget(Label(text="SR", size_hint=(None,None), size=(40,28)))
        self.sp_sr = Spinner(text=str(DEFAULT_SR), values=('44100','48000'), size_hint=(None,None), size=(100,28)); self.sp_sr.bind(text=lambda *a: self.on_change())
        self.add_widget(self.sp_sr)

        # Row 2
        self.add_widget(Label(text="FFT", size_hint=(None,None), size=(60,28)))
        self.sp_fft = Spinner(text=str(DEFAULT_FFT), values=('1024','2048','4096','8192'), size_hint=(None,None), size=(100,28)); self.sp_fft.bind(text=lambda *a: self.on_change())
        self.add_widget(self.sp_fft)

        self.add_widget(Label(text="Hop", size_hint=(None,None), size=(60,28)))
        self.sp_hop = Spinner(text=str(DEFAULT_HOP), values=('256','512','1024','2048'), size_hint=(None,None), size=(100,28)); self.sp_hop.bind(text=lambda *a: self.on_change())
        self.add_widget(self.sp_hop)

        self.add_widget(Label(text="WF Rows", size_hint=(None,None), size=(90,28)))
        self.sp_rows = Spinner(text=str(DEFAULT_WF_ROWS), values=('200','300','400','600'), size_hint=(None,None), size=(100,28)); self.sp_rows.bind(text=lambda *a: self.on_change())
        self.add_widget(self.sp_rows)

        self.add_widget(Label(text="Averaging", size_hint=(None,None), size=(100,28)))
        self.sl_smooth = Slider(min=0.0, max=0.98, value=DEFAULT_SMOOTH_ALPHA, step=0.02, size_hint=(None,None), size=(200,28)); self.sl_smooth.bind(value=lambda *a: self.on_change())
        self.add_widget(self.sl_smooth)

        # Row 3
        self.add_widget(Label(text="dB Floor", size_hint=(None,None), size=(80,28)))
        self.t_floor = TextInput(text=str(int(DEFAULT_DB_FLOOR)), multiline=False, size_hint=(None,None), size=(80,28)); self.t_floor.bind(text=lambda *a: self.on_change())
        self.add_widget(self.t_floor)

        self.add_widget(Label(text="dB Ceil", size_hint=(None,None), size=(80,28)))
        self.t_ceil = TextInput(text=str(int(DEFAULT_DB_CEIL)), multiline=False, size_hint=(None,None), size=(80,28)); self.t_ceil.bind(text=lambda *a: self.on_change())
        self.add_widget(self.t_ceil)

        self.add_widget(Label(text="Peak-Hold", size_hint=(None,None), size=(100,28)))
        self.cb_peak = CheckBox(active=True); self.cb_peak.bind(active=lambda *a: self.on_change()); self.add_widget(self.cb_peak)

        self.add_widget(Label(text="Decay dB/s", size_hint=(None,None), size=(110,28)))
        self.sl_decay = Slider(min=0, max=12, value=DEFAULT_PEAK_DECAY_DB_PER_S, size_hint=(None,None), size=(200,28)); self.sl_decay.bind(value=lambda *a: self.on_change())
        self.add_widget(self.sl_decay)

        # Row 4 (Filters & SPL & Actions)
        self.add_widget(Label(text="Filter", size_hint=(None,None), size=(80,28)))
        self.sp_filt = Spinner(text='None', values=('None','Bandpass','Notch'), size_hint=(None,None), size=(120,28)); self.sp_filt.bind(text=lambda *a: self.on_change())
        self.add_widget(self.sp_filt)

        self.add_widget(Label(text="f0 [Hz]", size_hint=(None,None), size=(80,28)))
        self.t_f0 = TextInput(text="1000", multiline=False, size_hint=(None,None), size=(80,28)); self.t_f0.bind(text=lambda *a: self.on_change())
        self.add_widget(self.t_f0)

        self.add_widget(Label(text="Q", size_hint=(None,None), size=(40,28)))
        self.t_Q = TextInput(text="3.0", multiline=False, size_hint=(None,None), size=(60,28)); self.t_Q.bind(text=lambda *a: self.on_change())
        self.add_widget(self.t_Q)

        self.add_widget(Label(text="SPL Offset [dB]", size_hint=(None,None), size=(130,28)))
        self.t_spl = TextInput(text="0.0", multiline=False, size_hint=(None,None), size=(90,28)); self.t_spl.bind(text=lambda *a: self.on_change())
        self.add_widget(self.t_spl)

        self.btn_pause = ToggleButton(text="Pause", state='normal', size_hint=(None,None), size=(90,32))
        self.btn_pause.bind(state=lambda *a: self.on_change()); self.add_widget(self.btn_pause)

        self.btn_csv = Button(text="Save CSV", size_hint=(None,None), size=(110,32))
        self.btn_csv.bind(on_press=lambda *a: self.on_save_csv()); self.add_widget(self.btn_csv)

        self.btn_shot = Button(text="Screenshot", size_hint=(None,None), size=(110,32))
        self.btn_shot.bind(on_press=lambda *a: self.on_screenshot()); self.add_widget(self.btn_shot)

    def get_values(self):
        def ffloat(t, dv):
            try: return float(t)
            except: return dv
        def fint(t, dv):
            try: return int(t)
            except: return dv
        return dict(
            colormap=self.sp_cmap.text,
            show_axes=self.cb_axes.active,
            mode=self.sp_mode.text,
            sr=fint(self.sp_sr.text, DEFAULT_SR),
            fft_size=fint(self.sp_fft.text, DEFAULT_FFT),
            hop_size=fint(self.sp_hop.text, DEFAULT_HOP),
            wf_rows=fint(self.sp_rows.text, DEFAULT_WF_ROWS),
            smooth_alpha=float(self.sl_smooth.value),
            db_floor=ffloat(self.t_floor.text, DEFAULT_DB_FLOOR),
            db_ceil=ffloat(self.t_ceil.text, DEFAULT_DB_CEIL),
            peak_enabled=self.cb_peak.active,
            peak_decay=float(self.sl_decay.value),
            filt_type=self.sp_filt.text,
            filt_f0=ffloat(self.t_f0.text, 1000.0),
            filt_Q=ffloat(self.t_Q.text, 3.0),
            spl_offset_db=ffloat(self.t_spl.text, 0.0),
            paused=(self.btn_pause.state=='down')
        )

# ---------- Root UI ----------
class RootUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        self.controls = ControlPanel(on_change=self.apply_options, on_save_csv=self.save_csv, on_screenshot=self.screenshot)
        self.add_widget(self.controls)

        self.n_bins = (DEFAULT_FFT//2)+1
        self.spectrum = Spectrum(size_hint=(1,0.42))
        self.spectrum.set_range(DEFAULT_DB_FLOOR, DEFAULT_DB_CEIL)
        self.spectrum.set_freq_axis(DEFAULT_SR, DEFAULT_FFT)
        self.waterfall = Waterfall(n_bins=self.n_bins, rows=DEFAULT_WF_ROWS, colormap='inferno', size_hint=(1,0.58))
        self.add_widget(self.spectrum); self.add_widget(self.waterfall)

        self.worker = AudioWorker(self.on_audio_frame)
        Clock.schedule_once(lambda dt: self.worker.start(), 0.5)
        Clock.schedule_once(lambda dt: self.apply_options(), 0.6)

        self.last_spec = None  # for CSV export

    def apply_options(self, *a):
        v = self.controls.get_values()

        # Pause
        self.worker.pause(v['paused'])

        # Spectrum visuals
        self.spectrum.set_axes(v['show_axes'])
        self.spectrum.set_range(v['db_floor'], v['db_ceil'])

        # Waterfall colormap
        self.waterfall.set_colormap(v['colormap'])

        need_restart = (
            self.worker.sr!=v['sr'] or
            self.worker.fft_size!=v['fft_size'] or
            self.worker.hop_size!=v['hop_size'] or
            self.worker.band_mode!=(v['mode'] if v['mode']!='FFT' else 'FFT') or
            True  # restart if filter or spl offset change requires reinit?
        )
        # live params (no restart required)
        self.worker.db_floor=v['db_floor']; self.worker.db_ceil=v['db_ceil']
        self.worker.smooth_alpha=v['smooth_alpha']
        self.worker.peak_decay_db_s = v['peak_decay'] if v['peak_enabled'] else 0.0
        self.worker.spl_offset_db = v['spl_offset_db']

        if need_restart:
            # rebuild WF if width changed
            if v['mode']=='FFT':
                n_bins = v['fft_size']//2 + 1
            else:
                bm = bins_for_bands(sr=v['sr'], nfft=v['fft_size'],
                                    mode=('Octave' if v['mode']=='Octave' else 'Third-Octave'))
                n_bins = len(bm) if bm else v['fft_size']//2 + 1
            if self.n_bins != n_bins or self.waterfall.rows != v['wf_rows']:
                self.n_bins = n_bins
                self.remove_widget(self.waterfall)
                self.waterfall = Waterfall(n_bins=self.n_bins, rows=v['wf_rows'], colormap=v['colormap'], size_hint=(1,0.58))
                self.add_widget(self.waterfall)

            # freq axis
            if v['mode']=='FFT':
                self.spectrum.set_freq_axis(v['sr'], v['fft_size'])
            else:
                bm = bins_for_bands(sr=v['sr'], nfft=v['fft_size'],
                                    mode=('Octave' if v['mode']=='Octave' else 'Third-Octave'))
                if bm:
                    centers=[fc for (_b0,_b1,fc) in bm]
                    self.spectrum.freq_bins_hz = np.array(centers, dtype=np.float32)
                else:
                    self.spectrum.freq_bins_hz = None

            self.worker.restart(sr=v['sr'], fft_size=v['fft_size'], hop_size=v['hop_size'],
                                band_mode=(v['mode'] if v['mode']!='FFT' else 'FFT'),
                                filt_type=v['filt_type'], filt_f0=v['filt_f0'], filt_Q=v['filt_Q'])
        else:
            if self.waterfall.rows != v['wf_rows']:
                self.remove_widget(self.waterfall)
                self.waterfall = Waterfall(n_bins=self.n_bins, rows=v['wf_rows'], colormap=v['colormap'], size_hint=(1,0.58))
                self.add_widget(self.waterfall)

    def on_audio_frame(self, spec_db, row_u8, peak_db, marker_txt):
        self.last_spec = spec_db
        # update spectrum + marker
        self.spectrum.set_spectrum(spec_db, peak_db, marker_text=marker_txt)
        # update waterfall
        self.waterfall.push_row_u8(row_u8)

    def save_csv(self):
        # Save current spectrum to CSV in app directory
        if self.last_spec is None: return
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(App.get_running_app().user_data_dir if hasattr(App.get_running_app(),'user_data_dir') else ".", f"spectrum_{ts}.csv")
        np.savetxt(path, self.last_spec, fmt="%.3f", delimiter=",")
        print(f"Saved CSV: {path}")

    def screenshot(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = f"screenshot_{ts}.png"
        # Save to current working dir
        Window.screenshot(name=name)
        print(f"Saved screenshot: {os.path.abspath(name)}")

    def stop(self): self.worker.stop()

class SpectrumApp(App):
    def build(self):
        self.title="Spectro Waterfall (Options)"
        Window.clearcolor=(0,0,0,1)
        self.root_ui = RootUI()
        return self.root_ui
    def on_stop(self):
        if hasattr(self,'root_ui'): self.root_ui.stop()

if __name__=="__main__":
    SpectrumApp().run()
