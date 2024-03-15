// cpp_src/my_application.cpp
#include "my_application.h"
#include <pybind11/embed.h>

namespace py = pybind11;

MyApplication::MyApplication() {
    // Constructor implementation
}

void MyApplication::runPythonCode() {
    py::scoped_interpreter guard{};  // Start the Python interpreter

    // py::module_ myModule = py::module_::import("my_python_module");
    // myModule.attr("my_function")();  // Replace with the actual function name

    py::print("Hello, World!"); // use the Python API
}
