import time
import threading
from typing import Optional, Callable


class StageTimer:
    def __init__(self):
        self._start_time: Optional[float] = None
        self._elapsed_ms: int = 0
        self._running = False
        self._paused = False
        self._pause_depth = 0
        self._lock = threading.Lock()
        self._on_pause_callbacks: list[Callable] = []
        self._on_resume_callbacks: list[Callable] = []

    def start(self, offset_ms: float = 0.0):
        with self._lock:
            self._start_time = time.perf_counter() - offset_ms / 1000.0
            self._elapsed_ms = 0
            self._running = True
            self._paused = False
            self._pause_depth = 0

    def pause(self):
        with self._lock:
            if not self._running or self._start_time is None:
                return
            self._pause_depth += 1
            if self._pause_depth == 1:
                now = time.perf_counter()
                self._elapsed_ms += int((now - self._start_time) * 1000)
                self._paused = True
                for cb in self._on_pause_callbacks:
                    cb()

    def resume(self):
        with self._lock:
            if not self._running:
                return
            self._pause_depth = max(0, self._pause_depth - 1)
            if self._pause_depth == 0 and self._paused:
                self._start_time = time.perf_counter()
                self._paused = False
                for cb in self._on_resume_callbacks:
                    cb()

    def toggle_pause(self):
        if self._paused:
            self.resume()
        else:
            self.pause()

    def get_elapsed_ms(self) -> int:
        with self._lock:
            if not self._running:
                return self._elapsed_ms
            if self._paused:
                return self._elapsed_ms
            now = time.perf_counter()
            return self._elapsed_ms + int((now - self._start_time) * 1000)

    def reset(self):
        with self._lock:
            self._start_time = None
            self._elapsed_ms = 0
            self._running = False
            self._paused = False
            self._pause_depth = 0

    def adjust(self, offset_ms: float):
        with self._lock:
            self._elapsed_ms += int(offset_ms)

    def register_pause_callback(self, callback: Callable):
        self._on_pause_callbacks.append(callback)

    def register_resume_callback(self, callback: Callable):
        self._on_resume_callbacks.append(callback)
