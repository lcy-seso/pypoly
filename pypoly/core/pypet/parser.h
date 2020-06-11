#ifndef PYPOLY_CORE_PYPET_PARSER_H_
#define PYPOLY_CORE_PYPET_PARSER_H_

#include "pypoly/core/pypet/array.pb.h"
#include "pypoly/core/pypet/ir_emitter.h"
#include "pypoly/core/pypet/pypet.h"

#include <isl/arg.h>
#include <isl/ctx.h>
#include <isl/options.h>
#include <pybind11/pybind11.h>
#include <string.h>  // for strdup
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/tree_views.h>

#include <map>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace pypoly {
namespace pypet {

using TorchDef = torch::jit::Def;
struct PypetScop;

struct TorchParser {
  // TODO(Ying) for experiment with TS parser only. Parsing is a recursive
  // process. Not implemented yet.
  TorchParser(std::string src)
      : src(std::move(src)), filename(""), file_lineno(0) {}
  explicit TorchParser(std::string src, std::string filename,
                       size_t file_lineno)
      : src(std::move(src)), filename(filename), file_lineno(file_lineno) {}

  TorchDef Parse() {
    torch::jit::Parser p(
        std::make_shared<torch::jit::Source>(src, filename, file_lineno));
    auto ast = TorchDef(p.parseFunction(/*is_method=*/true));
    return ast;
  }

  std::string src;
  std::string filename;
  // start line of the function to be parsed in the original file.
  size_t file_lineno;
};

class ParserImpl {
 public:
  explicit ParserImpl(const TorchDef& def, const ContextDesc& ctx_desc)
      : ast_(std::move(def)), ctx_desc_(std::move(ctx_desc)) {
    for (size_t i = 0; i < ctx_desc.vars_size(); ++i) {
      ctx_vars_.insert(std::make_pair(ctx_desc_.vars()[i].name(), i));
    }
  };

  PypetScop* ParseFunction();

 private:
  const TorchDef ast_;
  const ContextDesc ctx_desc_;

  // key is variable name in the codes; value is the position in `ctx_desc_`.
  std::map<std::string, size_t> ctx_vars_;

  // Traverse the AST to check whether there are SCoP(s) that could be
  // optimized.
  bool CheckScop();

  bool ParseDecl(isl_ctx* ctx);
  std::vector<PypetTree*> ParseBody(isl_ctx* ctx);
};

struct ScopParser {
  explicit ScopParser(const TorchDef& def)
      : pImpl(std::make_unique<ParserImpl>(def, ContextDesc())){};
  explicit ScopParser(const TorchDef& def, const ContextDesc& ctx_desc)
      : pImpl(std::make_unique<ParserImpl>(def, ctx_desc)){};

  ~ScopParser() = default;

  std::shared_ptr<PypetScop> Parse();

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

}  // namespace pypet
}  // namespace pypoly
#endif  // PYPOLY_CORE_PYPET_PARSER_H_
