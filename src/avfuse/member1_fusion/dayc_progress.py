"""Operational progress only. No scoring, cache rewriting, attempt control or retries."""
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
import threading
import time


class ProgressReporter:
    def __init__(self,interval=30,output=print):
        if interval<=0:raise ValueError('Positive heartbeat interval required')
        self.interval=interval;self.output=output;self.path=None;self.state={};self.lock=threading.Lock()
        self.stop=threading.Event();self.started=time.monotonic();self.thread=None

    def emit(self,event,**fields):
        with self.lock:
            record={'event':event,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds':round(time.monotonic()-self.started,3),**self.state,**fields}
            text=json.dumps(record,sort_keys=True,allow_nan=False)
            if self.path is not None:
                with self.path.open('a') as stream:stream.write(text+'\n');stream.flush()
            self.output(text)

    def update(self,**fields):
        with self.lock:self.state.update(fields)

    def attach(self,path):
        path=Path(path)
        with path.open('x'):pass
        self.path=path;self.emit('PROGRESS_LOG_OPENED')

    def __enter__(self):
        def heartbeat():
            while not self.stop.wait(self.interval):self.emit('HEARTBEAT')
        self.thread=threading.Thread(target=heartbeat,name='AFML_progress_only',daemon=True)
        self.thread.start();return self

    def __exit__(self,kind,value,traceback):
        self.stop.set()
        if self.thread is not None:self.thread.join()
        self.emit('OPERATION_INTERRUPTED' if kind is not None else 'OPERATIONS_FINISHED',
            error_type=kind.__name__ if kind is not None else None)
        return False

    @contextmanager
    def phase(self,name,**details):
        self.update(phase=name,**details);started=time.monotonic();self.emit('PHASE_STARTED')
        try:yield
        except BaseException as exc:
            self.emit('PHASE_FAILED',phase_seconds=round(time.monotonic()-started,3),error_type=type(exc).__name__);raise
        else:self.emit('PHASE_COMPLETED',phase_seconds=round(time.monotonic()-started,3))


def fresh_review_from_stdout(stdout,root,expected):
    """Only a new successful archive-emission marker may trigger a download."""
    matches=[line[len('RETURN REVIEW ZIP: '):].strip() for line in stdout.splitlines() if line.startswith('RETURN REVIEW ZIP: ')]
    if not matches:return None
    if len(matches)!=1:raise ValueError('Ambiguous current invocation review marker')
    path=Path(matches[0]);root=Path(root).resolve();expected=Path(expected).resolve()
    if path.is_symlink() or path.resolve()!=expected or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('Invalid current invocation review path')
    return path
