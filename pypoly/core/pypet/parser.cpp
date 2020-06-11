#include "pypoly/core/pypet/parser.h"

#include "pypoly/core/pypet/tree.h"
#include "pypoly/core/pypet/tree2scop.h"

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

  m.def("parse_scop", [](const std::string& src, const std::string& filename,
                         size_t file_lineno) {
    TorchParser p(src, filename, file_lineno);

    ScopParser scop_parser(p.Parse());
    scop_parser.Parse();
  });
}

void ScopParser::Parse() { pImpl->ParseFunction(); }

void ParserImpl::ParseDecl(isl_ctx* ctx) { LOG(INFO) << ast_.name(); }

std::vector<PypetTree*> ParserImpl::ParseBody(isl_ctx* ctx) {
  EmitStatements emitter(ctx, parsed_data_);
  return emitter(ast_.statements());
}

bool ParserImpl::CheckScop() {
  // TODO(Ying): Check whether SCoP is detected. Not implmented yet.
  return ast_.statements()[0].kind() == torch::jit::TK_FOR;
}

PypetScop* ParserImpl::ParseFunction() {
  LOG(INFO) << ast_.statements();
  if (!CheckScop()) {
    LOG(INFO) << "No SCoP is detected.";
    return nullptr;
  }

  struct isl_options* options = isl_options_new_with_defaults();
  isl_ctx* ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

  ParseDecl(ctx);
  std::vector<PypetTree*> trees = ParseBody(ctx);
  CHECK(trees.size() == 1U);

  TreeToScop converter(ctx);
  parsed_data_ = converter.ScopFromTree(trees[0]);

  isl_ctx_free(ctx);
  return parsed_data_;
}

}  // namespace pypet
}  // namespace pypoly
