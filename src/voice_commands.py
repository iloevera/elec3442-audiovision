import queue
import threading
import sounddevice as sd
import json

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    Model = None
    KaldiRecognizer = None

class VoskCommandListener:
    """
    Listens for a fixed set of voice commands using Vosk and a microphone.
    Use poll_command() to get the latest recognized command (returns None if nothing new).
    """
    def __init__(self, model_path, commands, sample_rate=16000, block_size=8000, input_device=None):
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("vosk is not installed. Install it with: pip install vosk")
        self._model_path = model_path
        self._commands = tuple(sorted({str(cmd).strip().lower() for cmd in commands if str(cmd).strip()}))
        if not self._commands:
            raise ValueError("At least one command must be provided")
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._input_device = input_device
        self._audio_queue = queue.Queue(maxsize=64)
        self._recognized_commands = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._stream = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        grammar = json.dumps(self._commands)
        model = Model(self._model_path)
        recognizer = KaldiRecognizer(model, self._sample_rate, grammar)
        def on_audio(indata, frames, time_info, status):
            if self._stop_event.is_set():
                raise sd.CallbackStop()
            self._audio_queue.put(bytes(indata))
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._block_size,
            callback=on_audio,
            device=self._input_device,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._run_recognition_loop, args=(recognizer,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._audio_queue.mutex:
            self._audio_queue.queue.clear()
        with self._recognized_commands.mutex:
            self._recognized_commands.queue.clear()

    def poll_command(self):
        try:
            return self._recognized_commands.get_nowait()
        except queue.Empty:
            return None

    def _run_recognition_loop(self, recognizer):
        while not self._stop_event.is_set():
            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = self._extract_text(result)
                cmd = self._match_command(text)
                if cmd:
                    self._recognized_commands.put(cmd)
            else:
                partial = recognizer.PartialResult()
                text = self._extract_text(partial)
                cmd = self._match_command(text)
                if cmd:
                    self._recognized_commands.put(cmd)

    @staticmethod
    def _extract_text(payload):
        try:
            obj = json.loads(payload)
            return obj.get("text", "") or obj.get("partial", "")
        except Exception:
            return ""

    def _match_command(self, text):
        text = text.lower().strip()
        for cmd in self._commands:
            if cmd in text:
                return cmd
        return None
