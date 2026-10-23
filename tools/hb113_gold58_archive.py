"""Read-only ZIP filesystem adapter for the unchanged P956 study_lut decoder.

Only requested Gold DICOM members are exposed. Bytes live until their decoder
file descriptor closes; no image cache or extraction is written to disk.
"""
import fnmatch
import threading
import types
import zipfile


class ArchiveReader:
    root = 'archive/train_series'

    def __init__(self, reader, path, ids):
        self.archive = zipfile.ZipFile(path)
        self.members = {}
        for info in self.archive.infolist():
            parts = info.filename.split('/')
            if (len(parts) == 4 and parts[0] == 'train_series' and
                    parts[1] in ids and parts[3].endswith('.dcm')):
                name = 'archive/' + info.filename
                if name in self.members:
                    raise ValueError('duplicate DICOM archive member')
                self.members[name] = info
        if {n.split('/')[2] for n in self.members} != set(ids):
            self.archive.close()
            raise ValueError('archive missing requested Gold studies')
        self.handles = {}
        self.next_fd = 0
        self.read_errors = 0
        self.lock = threading.Lock()
        reader.glob = types.SimpleNamespace(glob=self.glob)
        reader._fopen = self.open
        reader._pread = self.pread
        reader.os = types.SimpleNamespace(fstat=self.fstat, close=self.close_fd)

    def glob(self, pattern):
        # Preserve archive directory order; P956 then sorts by DICOM position.
        return [name for name in self.members if fnmatch.fnmatchcase(name, pattern)]

    def open(self, path, fadvise_random=False):
        with self.lock:
            # Serialize seek/read on the shared archive handle (including CRC).
            try:
                data = self.archive.read(self.members[path])
            except Exception:
                self.read_errors += 1
                raise
            self.next_fd += 1
            self.handles[self.next_fd] = data
            return self.next_fd

    def pread(self, fd, n, offset):
        return self.handles[fd][offset:offset+n]

    def fstat(self, fd):
        return types.SimpleNamespace(st_size=len(self.handles[fd]))

    def close_fd(self, fd):
        with self.lock:
            del self.handles[fd]

    def close(self):
        self.archive.close()
        if self.handles:
            raise ValueError('decoder leaked archive file descriptors')
