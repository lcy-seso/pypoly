#include "pypet/core/parser.h"

namespace pypet {

PYBIND11_MODULE(_parser, m) {
  m.def("parse_scop", [](const std::string& src) {
    TorchParser p(src);
    ScopParser scop_parser(p.Parse());
    scop_parser.DumpAST();
  });

  py::class_<PypetScop>(m, "PypetScop").def(py::init());
}
}  // namespace pypet
