#include "pypoly/core/pypet/parser.h"

#include "pypoly/core/pypet/array.h"
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

  py::class_<torch::jit::Def>(m, "Def").def("__repr__",
                                            [](const torch::jit::Def& self) {
                                              std::ostringstream s;
                                              s << self;
                                              return s.str();
                                            });

  m.def("parse_scop",
        [](const torch::jit::Def& ast, const std::string& contexts) {
          ContextDesc ctx_desc;
          bool success = ctx_desc.ParseFromString(contexts);
          CHECK(success) << "Fail to parse context variables." << std::endl;

          ScopParser scop_parser(ast, ctx_desc);
          *scop_parser.Parse();
        });

  m.def("get_torch_ast", [](const std::string& src, const std::string& filename,
                            size_t file_lineno) {
    TorchParser p(src, filename, file_lineno);
    return p.Parse();
  });
}

std::shared_ptr<PypetScop> ScopParser::Parse() {
  // TODO(Ying): bad practice that does not use make_shared to create a smart
  // pointer.
  std::shared_ptr<PypetScop> parsed_data(pImpl->ParseFunction());
  return parsed_data;
}

bool ParserImpl::ParseDecl(isl_ctx* ctx) {
  LOG(INFO) << ast_.name();
  return true;
}

std::vector<PypetTree*> ParserImpl::ParseBody(isl_ctx* ctx) {
  EmitStatements emitter(ctx);
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
  // Torch JIT AST to PypetTree representation.
  // trees owns pointers pointing to PypetTree object.
  std::vector<PypetTree*> trees = ParseBody(ctx);
  CHECK(trees.size() == 1U);

  TreeToScop converter(ctx);
  PypetScop* scop = converter.ScopFromTree(trees[0]);

  // TODO(Ying): Current implementation does not distinguish variable
  // declaration and name reuse, so integers defined in the scop is
  // not able to be recognized as arrays.
  ArrayScanner scanner(&ctx_desc_);
  scop = scanner.ScanArrays(ctx, scop);

  std::cout << scop << std::endl;

  isl_ctx_free(ctx);
  return scop;
}

}  // namespace pypet
}  // namespace pypoly
