"""Time one module load in one fresh CUDA process."""
import argparse
import ctypes as C
import json
from pathlib import Path
from time import perf_counter_ns

U32 = C.c_uint32
DEVICE_POINTER = C.c_uint64
HANDLE = C.c_void_p
GUARD = 0xA5A5A5A5


def input_values(n):
    return [(i * 2654435761 + 17) & 0xFFFFFFFF for i in range(n)]


def oracle(values):
    return [(value + value + value + (i % 17)) % (2 ** 32)
            for i, value in enumerate(values)]


def validate_output(expected, actual):
    if actual != expected + [GUARD]:
        raise ValueError('output or tail guard mismatch')


class Driver:
    def __init__(self):
        self.library = C.CDLL('libcuda.so.1')

    def call(self, name, types, *values, expected=0):
        function = getattr(self.library, name)
        function.argtypes = types
        function.restype = C.c_int
        result = function(*values)
        if result != expected:
            raise RuntimeError(f'{name}: CUDA error {result}, expected {expected}')


def run(image, n, negative):
    cuda = Driver()
    call = cuda.call
    device, context = C.c_int(), HANDLE()
    call('cuInit', [C.c_uint], 0)
    call('cuDeviceGet', [C.POINTER(C.c_int), C.c_int], C.byref(device), 0)
    name = C.create_string_buffer(100)
    call('cuDeviceGetName', [HANDLE, C.c_int, C.c_int], name, len(name), device)
    major, minor, driver = C.c_int(), C.c_int(), C.c_int()
    for attr, output in [(75, major), (76, minor)]:
        call('cuDeviceGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_int],
             C.byref(output), attr, device)
    if 'B200' not in name.value.decode() or (major.value, minor.value) != (10, 0):
        raise RuntimeError('NVIDIA B200 compute capability 10.0 required')
    call('cuDriverGetVersion', [C.POINTER(C.c_int)], C.byref(driver))
    call('cuCtxCreate_v2', [C.POINTER(HANDLE), C.c_uint, C.c_int],
         C.byref(context), 0, device)
    try:
        module, kernel = HANDLE(), HANDLE()
        binary = C.create_string_buffer(image.read_bytes())
        start = perf_counter_ns()
        call('cuModuleLoadData', [C.POINTER(HANDLE), HANDLE], C.byref(module), binary)
        loaded = perf_counter_ns()
        if negative:
            # CUDA_ERROR_NOT_FOUND from a valid module and a nonexistent entry point.
            call('cuModuleGetFunction', [C.POINTER(HANDLE), HANDLE, C.c_char_p],
                 C.byref(kernel), module, b'absent_kernel', expected=500)
        call('cuModuleGetFunction', [C.POINTER(HANDLE), HANDLE, C.c_char_p],
             C.byref(kernel), module, b'transform')
        resolved = perf_counter_ns()
        values = input_values(n)
        source = (U32 * n)(*values)
        target = (U32 * (n + 1))(*([GUARD] * (n + 1)))
        src, dst = DEVICE_POINTER(), DEVICE_POINTER()
        for pointer, host in [(src, source), (dst, target)]:
            call('cuMemAlloc_v2', [C.POINTER(DEVICE_POINTER), C.c_size_t],
                 C.byref(pointer), C.sizeof(host))
            call('cuMemcpyHtoD_v2', [DEVICE_POINTER, HANDLE, C.c_size_t],
                 pointer, host, C.sizeof(host))
        length = U32(n)
        args = (HANDLE * 3)(*(C.cast(C.byref(x), HANDLE) for x in (src, dst, length)))
        launch_start = perf_counter_ns()
        call('cuLaunchKernel', [HANDLE] + [C.c_uint] * 7 + [HANDLE] * 3,
             kernel, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, args, None)
        call('cuCtxSynchronize', [])
        completed = perf_counter_ns()
        call('cuMemcpyDtoH_v2', [HANDLE, DEVICE_POINTER, C.c_size_t],
             target, dst, C.sizeof(target))
        validate_output(oracle(values), list(target))
        return dict(gpu=name.value.decode(), driver_api=driver.value, n=n,
                    bytes=image.stat().st_size, module_ns=loaded-start,
                    lookup_ns=resolved-loaded, launch_sync_ns=completed-launch_start)
    finally:
        call('cuCtxDestroy_v2', [HANDLE], context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('n', type=int)
    parser.add_argument('--negative', action='store_true')
    options = parser.parse_args()
    if not 1 <= options.n <= 65537:
        parser.error('n must be in [1, 65537]')
    print(json.dumps(run(options.image, options.n, options.negative)))


if __name__ == '__main__':
    main()
