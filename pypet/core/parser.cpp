#include "pypet/core/parser.h"

#include <torch/csrc/jit/frontend/parser.h>

namespace pypet {

PYBIND11_MODULE(_parser, m) {
  m.def("parse_scop", [](const std::string& src) {
    torch::jit::Parser p(std::make_shared<torch::jit::Source>(src));
    auto ast = torch::jit::Def(p.parseFunction(/*is_method=*/true));
    auto scop_parser = ScopParser(ast);
  });
}
}  // namespace pypet
