import ctypes, mmap, os, sys
f = sys.argv[1]
fd = os.open(f, os.O_RDONLY); size = os.fstat(fd).st_size
m = mmap.mmap(fd, size, access=mmap.ACCESS_COPY)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
buf_t = ctypes.c_char * ((size + 4095) // 4096)
vec = buf_t()
addr = ctypes.c_void_p.from_buffer(ctypes.c_char.from_buffer(m)) if False else None
base = ctypes.addressof(ctypes.c_char.from_buffer(m))
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
r = libc.mincore(ctypes.c_void_p(base), ctypes.c_size_t(size), ctypes.byref(vec))
res = sum(b & 1 for b in vec.raw)
print("rc", r, "resident pages", res, "of", len(vec.raw), "= %.1f%%" % (100.0 * res / len(vec.raw)), "= %.1f GiB" % (res * 4096 / 2**30))
