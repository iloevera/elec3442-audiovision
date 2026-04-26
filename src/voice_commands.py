from __future__ import annotations

from collections import deque
import json
import queue
import threading
from typing import Iterable

import sounddevice as sd

try:
    from vosk import KaldiRecognizer, Model
except ImportError:  # pragma: no cover - optional dependency
    KaldiRecognizer = None
    Model = None


class VoskCommandListener:
    """Background listener for short keyword-style voice commands."""

    def __init__(
        self,
        *,
        model_path: str,
        commands: Iterable[str],
        sample_rate: int = 16_000,
        block_size: int = 8_000,
        input_device: int | None = None,
    ) -> None:
        if Model is None or KaldiRecognizer is None:
            raise RuntimeError("vosk is not installed. Install it with: pip install vosk")

        self._model_path = str(model_path)
        self._commands = tuple(sorted({str(command).strip().lower() for command in commands if str(command).strip()}))
        if not self._commands:
            raise ValueError("At least one command must be provided")

        self._sample_rate = int(sample_rate)
        self._block_size = int(block_size)
        self._input_device = input_device

        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._recognized_commands: deque[str] = deque()
        self._recognized_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: sd.RawInputStream | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._stop_event.clear()
        grammar = json.dumps(self._commands)
        model = Model(self._model_path)
        recognizer = KaldiRecognizer(model, self._sample_rate, grammar)

        def on_audio(indata, frames, time_info, status) -> None:  # noqa: ANN001
            del frames, time_info, status
            if self._stop_event.is_set():
                return
            try:
                self._audio_queue.put_nowait(bytes(indata))
            except queue.Full:
                # Avoid blocking the audio callback if recognition falls behind.
                pass

        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._block_size,
            callback=on_audio,
            device=self._input_device,
        )
        self._stream.start()

        self._thread = threading.Thread(
            target=self._run_recognition_loop,
            args=(recognizer,),
            name="VoskCommandListener",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=1.0)

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def poll_command(self) -> str | None:
        with self._recognized_lock:
            if not self._recognized_commands:
                return None
            return self._recognized_commands.popleft()

    def _run_recognition_loop(self, recognizer) -> None:  # noqa: ANN001
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.15)
            except queue.Empty:
                continue

            if recognizer.AcceptWaveform(chunk):
                result_text = self._extract_text(recognizer.Result())
            else:
                result_text = self._extract_text(recognizer.PartialResult())

            if not result_text:
                continue

            command = self._match_command(result_text)
            if command is None:
                continue

            with self._recognized_lock:
                self._recognized_commands.append(command)

    @staticmethod
    def _extract_text(payload: str) -> str:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return ""
        text = parsed.get("text") or parsed.get("partial") or ""
        return str(text).strip().lower()

    def _match_command(self, text: str) -> str | None:
        for command in self._commands:
            if command in text:
                return command
        return None
