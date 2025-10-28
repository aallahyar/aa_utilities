from datetime import datetime
from pathlib import Path
import gzip
import pickle
import os
import stat

class Checkpoint:
    def __init__(self, path=None, verbose=True):
        self.verbose = verbose
        if path is None:
            path = datetime.now().strftime('./.analyses_checkpoints/%Y%m%d_%H%M%S/')
        self.path = Path(path)
        existed = self.path.exists()
        self.path.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f'Checkpoint folder {"found" if existed else "created"} at: {self.path}')

    def save(self, obj, file_name='data.pkl.gz', overwrite=True):
        """Save an object to the checkpoint folder using gzip compression."""
        fpath = self.path / file_name
        if fpath.exists():
            if not overwrite:
                raise FileExistsError(f'File exists: {fpath}')
            else:
                if self.verbose:
                    print(f'Warning, destination file is overwritten: {fpath}')
        with gzip.open(fpath, mode='wb', compresslevel=5) as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get size after write
        if self.verbose:
            size_bytes = os.path.getsize(fpath)
            size_human = Checkpoint.human_readable_size(size_bytes)
            print(f'Successfully stored in: {fpath}  ({size_human})')

    def load(self, file_name='data.pkl.gz'):
        fpath = self.path / file_name
        try:
            with gzip.open(fpath, 'rb') as fh:
                if self.verbose:
                    size_bytes = os.path.getsize(fpath)
                    size_human = Checkpoint.human_readable_size(size_bytes)
                    print(f'Loading from: {fpath}  ({size_human})')
                obj = pickle.load(fh)
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found: {fpath}')
        else:
            if self.verbose:
                print(f'Successfully loaded from: {fpath}')
        return obj

    @staticmethod
    def human_readable_size(size, decimal_places=1):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0 or unit == 'TB':
                return f"{size:.{decimal_places}f} {unit}"
            size /= 1024.0

    def list(self, pattern='*.pkl.gz', extended=True):
        entries = []
        for fpath in self.path.glob(pattern):
            if not extended:
                entries.append(fpath)
                continue
            st = fpath.stat()
            mode = st.st_mode
            perms = ''.join([
                'd' if fpath.is_dir() else '-',
                'r' if mode & stat.S_IRUSR else '-',
                'w' if mode & stat.S_IWUSR else '-',
                'x' if mode & stat.S_IXUSR else '-',
                'r' if mode & stat.S_IRGRP else '-',
                'w' if mode & stat.S_IWGRP else '-',
                'x' if mode & stat.S_IXGRP else '-',
                'r' if mode & stat.S_IROTH else '-',
                'w' if mode & stat.S_IWOTH else '-',
                'x' if mode & stat.S_IXOTH else '-',
            ])
            mtime_str = datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            entries.append({
                'name': fpath.name,
                'path': fpath,
                'size': st.st_size,
                'size_human': self.human_readable_size(st.st_size),
                'mtime': st.st_mtime,
                'mtime_str': mtime_str,
                'perms': perms,
            })
        if self.verbose and extended:
            for e in entries:
                print(f"{e['perms']}  {e['size_human']:>10}  {e['mtime_str']:<20}  {e['name']}")
        return entries

    def __repr__(self):
        return f'{self.__class__.__name__}(path={self.path})'