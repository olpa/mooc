#include "./add_cuda.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cuda", &add_cuda, "Add two tensors (CUDA)");
}
