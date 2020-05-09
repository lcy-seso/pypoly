#include "pypet/core/parser.h"

namespace pypet {

std::once_flag glog_init_flag;

void InitGLOG(const std::string& prog_name) {
  std::call_once(glog_init_flag, [&]() {
    google::InitGoogleLogging(strdup(prog_name.c_str()));
  });
}

PYBIND11_MODULE(_parser, m) {
  m.def("init_glog", InitGLOG);

  m.def("parse_scop", [](const std::string& src) {
    TorchParser p(src);

    ScopParser scop_parser(p.Parse());
    // scop_parser.DumpAST();
    scop_parser.Parse();
  });

  py::class_<PypetScop>(m, "PypetScop").def(py::init());
}

void ScopParser::Parse() { pImpl->ParseFunction(); }

void ParserImpl::ParseDecl() { LOG(INFO) << ast_.name(); }

void ParserImpl::ParseBody() {
  EmitStatements emitter(std::make_shared<PypetScop>(parsed_data_));
  emitter(ast_.statements());
}

void ParserImpl::ParseFunction() {
  LOG(INFO) << ast_.statements();

  ParseDecl();
  ParseBody();
}

}  // namespace pypet
