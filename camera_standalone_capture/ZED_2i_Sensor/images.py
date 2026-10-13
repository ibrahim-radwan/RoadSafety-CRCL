"""Bounded image writes from owned SDK buffers, with explicit quality settings."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context, shared_memory
import signal
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import check_size

_worker_segments = {}


def image_filenames(config, index, stamp):
    extension = 'jpg' if config['images']['format'] == 'JPEG' else config['images']['format'].lower()
    return {s: f'images/{s}/{stamp}_{index:08d}.{extension}' for s in config['images']['streams']}


def worker_init():
    # The parent handles Ctrl+C and drains accepted frames. Spawn avoids
    # inheriting CUDA state or camera handles into image workers.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    Image.preinit()


def worker_ready():
    return True


def write_image_frame(root, config, index, stamp, host, arrays):
    rows = []
    for stream, filename in image_filenames(config, index, stamp).items():
        path = root / filename
        temporary = path.with_suffix(path.suffix + '.part')
        try:
            size = (config['camera']['width'], config['camera']['height'])
            if config['images']['format'] in ('BMP', 'JPEG'):
                options = {'quality': config['images']['jpeg_quality'], 'subsampling': 0,
                           'optimize': False} if config['images']['format'] == 'JPEG' else {}
                Image.frombytes('RGB', size, arrays[stream], 'raw', 'BGRX', 0, 1).save(
                    temporary, format=config['images']['format'], **options)
            else:
                # Dataset images need RGB; the SDK's opaque alpha byte is redundant.
                Image.frombytes('RGB', size, arrays[stream], 'raw', 'BGRX', 0, 1).save(
                    temporary, format='PNG', compress_level=config['images']['png_compression'])
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        rows.append({'frame_index': index, 'timestamp_ns': stamp, 'host_receipt_ns': host,
                     'stream': stream, 'file': filename,
                     'width': config['camera']['width'], 'height': config['camera']['height'],
                     'encoding': 'RGB8',
                     'pixel_lossless': config['images']['format'] != 'JPEG',
                     'jpeg_quality': config['images']['jpeg_quality'] if config['images']['format'] == 'JPEG' else '',
                     'source': 'SDK_unrectified_live' if stream.endswith('_raw') else 'SDK_rectified_live'})
    return rows


def write_shared_frame(root, config, index, stamp, host, name, specs):
    # Persistent mappings avoid per-frame mmap/page-fault overhead. Workers
    # close with the pool; only the parent owns/unlinks these bounded buffers.
    if name not in _worker_segments:
        _worker_segments[name] = shared_memory.SharedMemory(name=name)
    segment = _worker_segments[name]
    arrays = {stream: np.ndarray(shape,dtype=np.uint8,buffer=segment.buf,offset=offset)
              for stream,shape,offset in specs}
    try:
        return write_image_frame(root, config, index, stamp, host, arrays)
    finally:
        arrays.clear()


class ImageSink:
    def __init__(self, root, config, csv_sink):
        # Import/register encoders before capture, avoiding a first-frame stall
        # while Pillow lazily imports plugins with the Python GIL held.
        Image.preinit()
        self.root, self.config, self.csv_sink = root, config, csv_sink
        self.streams = tuple(config['images']['streams'])
        self.limit = config['images']['queue_frames']
        self.pending = deque()
        self.counts = Counter()
        self.high_water_frames = 0
        self.backpressure_events = 0
        self.backpressure_sec = 0.
        self.buffers = {}
        self.all_buffers = []
        self.free_buffers = deque()
        self.backend = config['images']['writer_backend']
        if self.backend == 'auto':
            self.backend = 'thread' if config['images']['format'] == 'BMP' else 'process'
        for stream in self.streams:
            (root / 'images' / stream).mkdir(parents=True, exist_ok=True)
        if self.backend == 'process':
            shared_bytes = config['camera']['width']*config['camera']['height']*4*len(self.streams)*(self.limit+1)
            if Path('/dev/shm').is_dir() and shutil.disk_usage('/dev/shm').free < shared_bytes:
                raise RuntimeError('Insufficient shared memory for images.queue_frames; reduce the queue or use thread writers')
            self.pool = ProcessPoolExecutor(max_workers=config['images']['workers'],
                mp_context=get_context('spawn'), initializer=worker_init)
            try:
                ready = [self.pool.submit(worker_ready) for _ in range(config['images']['workers'])]
                for future in ready:
                    future.result()
                frame_bytes = config['camera']['width']*config['camera']['height']*4*len(self.streams)
                for _ in range(self.limit):
                    segment = shared_memory.SharedMemory(create=True,size=frame_bytes)
                    self.all_buffers.append(segment)
                    # Touch pages before acquisition instead of faulting/zeroing
                    # newly allocated shared memory in the camera loop.
                    np.ndarray((frame_bytes,),dtype=np.uint8,buffer=segment.buf).fill(0)
                    self.free_buffers.append(segment)
            except BaseException:
                self.pool.shutdown(wait=True)
                for segment in self.all_buffers:
                    segment.close()
                    segment.unlink()
                raise
        else:
            self.pool = ThreadPoolExecutor(max_workers=config['images']['workers'],
                                           thread_name_prefix='zed-images')

    def filenames(self, index, stamp):
        return image_filenames(self.config, index, stamp)

    def check(self):
        while self.pending and self.pending[0].done():
            self._finish()
        # A later worker may fail before the oldest frame finishes.
        for future in self.pending:
            if future.done() and future.exception() is not None:
                future.result()

    def _finish(self):
        # Publish only completed frames, in submission order. Workers may finish
        # out of order; that must not reorder the timestamp/file index.
        future = self.pending.popleft()
        try:
            for row in future.result():
                self.csv_sink.put('images/frames.csv', row)
                self.counts[row['stream']] += 1
        finally:
            segment = self.buffers.pop(future, None)
            if segment is not None:
                self.free_buffers.append(segment)

    def prepare(self):
        """Apply backpressure before copying another frame, keeping memory bounded."""
        self.check()
        if len(self.pending) >= self.limit:
            started = time.monotonic()
            self.backpressure_events += 1
            self._finish()
            self.backpressure_sec += time.monotonic() - started

    def put(self, index, stamp, host, arrays):
        if set(arrays) != set(self.streams):
            raise ValueError('Direct image frame does not contain all configured streams')
        for array in arrays.values():
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 4:
                raise ValueError('Direct images must be BGRA8 SDK buffers')
            check_size(array.shape[1], array.shape[0], self.config)
            if not array.flags.c_contiguous:
                raise ValueError('Direct image workers require owned contiguous SDK copies')
        # Threads require owned copies. Processes synchronously stage SDK views
        # into a leased shared buffer before any next grab can reuse them.
        self.prepare()
        if self.backend == 'process':
            segment = self.free_buffers.popleft()
            try:
                offset, specs = 0, []
                for stream, array in arrays.items():
                    target = np.ndarray(array.shape,dtype=np.uint8,buffer=segment.buf,offset=offset)
                    target[:] = array
                    del target
                    specs.append((stream,array.shape,offset))
                    offset += array.nbytes
                future = self.pool.submit(write_shared_frame,self.root,self.config,index,stamp,host,segment.name,specs)
                self.buffers[future] = segment
            except BaseException:
                self.free_buffers.append(segment)
                raise
        else:
            future = self.pool.submit(self._write, index, stamp, host, arrays)
        self.pending.append(future)
        self.high_water_frames = max(self.high_water_frames, len(self.pending))

    def _write(self, index, stamp, host, arrays):
        return write_image_frame(self.root, self.config, index, stamp, host, arrays)

    def close(self):
        # Drain every accepted frame even if another worker failed.
        self.pool.shutdown(wait=True)
        error = None
        while self.pending:
            try:
                self._finish()
            except Exception as exc:
                error = error or exc
        for segment in self.all_buffers:
            try:
                segment.close()
                segment.unlink()
            except Exception as exc:
                error = error or exc
        self.all_buffers.clear()
        self.free_buffers.clear()
        if error is not None:
            raise RuntimeError('Direct image saving failed') from error

    def stats(self):
        return {'streams': dict(self.counts), 'max_pending_frames': self.high_water_frames,
                'backpressure_events': self.backpressure_events,
                'backpressure_sec': self.backpressure_sec,
                'queue_limit_frames': self.limit,
                'writer_backend': self.backend,
                'shared_buffer_slots': self.limit if self.backend == 'process' else 0,
                'encoding': {'BMP': 'lossless_RGB8_BMP', 'PNG': 'lossless_RGB8_PNG',
                             'JPEG': 'lossy_RGB8_JPEG_444'}[self.config['images']['format']],
                'jpeg_quality': self.config['images']['jpeg_quality'] if self.config['images']['format'] == 'JPEG' else None}
