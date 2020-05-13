#ifndef _PARSER_H
#define _PARSER_H

#include "pypoly/core/pypet/ir_emitter.h"
#include "pypoly/core/pypet/pypet.h"

#include <glog/logging.h>
#include <isl/arg.h>
#include <isl/ctx.h>
#include <isl/options.h>
#include <pybind11/pybind11.h>
#include <string.h>  // for strdup
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/tree_views.h>

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
  TorchParser(std::string src) : src(std::move(src)) {}

  TorchDef Parse() {
    torch::jit::Parser p(std::make_shared<torch::jit::Source>(src));
    auto ast = TorchDef(p.parseFunction(/*is_method=*/true));
    return ast;
  }

  std::string src;
};

class ParserImpl {
 public:
  explicit ParserImpl(const TorchDef& def)
      : ast_(std::move(def)), parsed_data_(PypetScop()){};

  void DumpAST() const { LOG(INFO) << ast_; }
  PypetScopPtr ParseFunction();

 private:
  TorchDef ast_;
  PypetScop parsed_data_;

  /* Traverse the AST to check whether there are SCoP(s) that could be
   * optimized. */
  bool CheckScop();

  void ParseDecl(isl_ctx* ctx);
  void ParseBody(isl_ctx* ctx);
};

struct ScopParser {
  explicit ScopParser(const TorchDef& def) : pImpl(new ParserImpl(def)) {}
  ~ScopParser() = default;

  void DumpAST() const { pImpl->DumpAST(); }
  void Parse();

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

}  // namespace pypet
}  // namespace pypoly
#endif
