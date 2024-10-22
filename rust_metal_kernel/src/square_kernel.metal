#include <metal_stdlib>
using namespace metal;

kernel void square_kernel(const device float *in [[buffer(0)]],
                          device float *out [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * in[id];
}

kernel void cube_kernel(const device float *in [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * in[id] * in[id];
}
