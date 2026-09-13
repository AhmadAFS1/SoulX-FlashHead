"""Single bounded IPC slot; parent copies before another generation can use it.

Returned RGB arrays own their storage: slow encoders and cancelled turns never
borrow a slot that a worker can overwrite. The creating parent owns unlinking.
"""
from multiprocessing.shared_memory import SharedMemory
import math
import numpy as np


class SharedChunks:
    def __init__(self, shape, name=None):
        self.shape=tuple(shape)
        if len(shape)!=5 or shape[-1]!=3 or any(not isinstance(v,int) or v<=0 for v in shape):
            raise ValueError('Expected positive B,T,H,W,3 shape')
        self.owner=name is None
        self.shm=SharedMemory(create=self.owner,size=math.prod(shape) if self.owner else 0,name=name)
        if self.shm.size < math.prod(shape):
            self.shm.close()
            raise ValueError('Shared chunk segment is too small')
        self.array=np.ndarray(self.shape,dtype=np.uint8,buffer=self.shm.buf)
        self.closed=False

    def spec(self):
        return dict(shape=self.shape,name=self.shm.name)

    def write(self,chunks):
        if self.closed or len(chunks)>self.shape[0]:
            raise ValueError('Invalid shared chunk batch')
        counts=[]
        for i,chunk in enumerate(chunks):
            if chunk.dtype!=np.uint8 or chunk.ndim!=4 or chunk.shape[1:]!=self.shape[2:] or not 0<len(chunk)<=self.shape[1]:
                raise ValueError('Invalid shared RGB chunk')
            np.copyto(self.array[i,:len(chunk)],chunk)
            counts.append(len(chunk))
        return counts

    def read(self,counts):
        if self.closed or len(counts)>self.shape[0] or any(type(c)!=int or not 0<c<=self.shape[1] for c in counts):
            raise ValueError('Invalid shared chunk descriptor')
        return [self.array[i,:n].copy() for i,n in enumerate(counts)]

    def close(self):
        if not self.closed:
            self.closed=True
            self.array=None
            self.shm.close()
            if self.owner:
                self.shm.unlink()
