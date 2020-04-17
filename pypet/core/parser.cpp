#include <pypet/core/parser.h>
namespace pypet {

PYBIND11_MODULE(_parser, m) {
  py::class_<ScopParser>(m, "ScopParser").def(py::init<const std::string&>());
}
}  // namespace pypet
