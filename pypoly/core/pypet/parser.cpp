#include "pypoly/core/pypet/parser.h"

namespace pypoly {
namespace pypet {

std::once_flag glog_init_flag;

void InitGLOG(const std::string& prog_name) {
  std::call_once(glog_init_flag, [&]() {
    // TODO
    // google::InitGoogleLogging(strdup(prog_name.c_str()));
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
}

void ScopParser::Parse() { pImpl->ParseFunction(); }

void ParserImpl::ParseDecl(isl_ctx* ctx) { LOG(INFO) << ast_.name(); }

void ParserImpl::ParseBody(isl_ctx* ctx) {
  EmitStatements emitter(ctx, std::make_shared<PypetScop>(parsed_data_));
  emitter(ast_.statements());
}

bool ParserImpl::CheckScop() {
  // TODO(Ying): Check whether SCoP is detected. Not implmented yet.
  return ast_.statements()[0].kind() == torch::jit::TK_FOR;
}

PypetScopPtr ParserImpl::ParseFunction() {
  LOG(INFO) << ast_.statements();
  if (!CheckScop()) {
    LOG(INFO) << "No SCoP is detected.";
    return nullptr;
  }

  struct isl_options* options = isl_options_new_with_defaults();
  isl_ctx* ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

  ParseDecl(ctx);
  ParseBody(ctx);

  isl_ctx_free(ctx);
  return std::make_shared<PypetScop>(parsed_data_);
}

}  // namespace pypet
}  // namespace pypoly
