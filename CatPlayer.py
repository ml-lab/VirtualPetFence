import threading
import wave

import pyaudio


class CatPlayer:
    def __init__(self):
        f = wave.open("cat.wav", "rb")
        self.f = f
        p = pyaudio.PyAudio()
        self.stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                channels=f.getnchannels(),
                rate=f.getframerate(),
                output=True)
        self.e = threading.Event()
        self.t = threading.Thread(target=self._play_cat_sound, args=(self.e,))
        self.t.daemon = True
        self.t.start()

    def _play_cat_sound(self, e):
        f, stream = self.f, self.stream
        chunk = 1024
        while True:
            f.setpos(0)
            data = f.readframes(chunk)
            while data != '':
                e.wait()
                stream.write(data)
                data = f.readframes(chunk)
            e.clear()

    def play(self):
        self.e.set()

    def pause(self):
        self.e.clear()