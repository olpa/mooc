#include <cuda_device_runtime_api.h>
#include "./vector_add.cu"

torch::Tensor add_cuda(torach::Tensor a, torch::Tensor b) {
    auto c = torch::empty_line(a);
    int n = a.nume1();
    int blocks = (n + 255) / 256;
    add_kernel<<<blocks, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}
