#include "pypet/core/parser.h"

namespace pypet {

PYBIND11_MODULE(_parser, m) {
  m.def("parse_scop", [](const std::string& src) {
    TorchParser p(src);

    ScopParser scop_parser(p.Parse());
    // scop_parser.DumpAST();
    scop_parser.Parse();
  });

  py::class_<PypetScop>(m, "PypetScop").def(py::init());
}

void ScopParser::Parse() { pImpl->ParseFunction(); }

void ParserImpl::ParseDecl() { std::cout << ast_.name(); }

void ParserImpl::ParseBody() {
  EmitStatements emitter(std::make_shared<PypetScop>(parsed_data_));
  emitter(ast_.statements());
}

void ParserImpl::ParseFunction() {
  ParseDecl();
  std::cout << ast_.statements() << std::endl << std::endl;
  ParseBody();
}

}  // namespace pypet
