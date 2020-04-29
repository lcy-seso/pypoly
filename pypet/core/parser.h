#ifndef _PARSER_H
#define _PARSER_H

#include "pypet/core/pypet.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/tree_views.h>

#include <vector>

namespace py = pybind11;

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
  void DumpAST() const { std::cout << ast_; }
  void ParseDecl();
  void ParseBody();
  void ParseFunction();

 private:
  void emitFor(const torch::jit::For& stmt);
  void emitIf(const torch::jit::If& stmt);
  void emitWhile(const torch::jit::While& stmt);
  void emitAssignment(const torch::jit::Assign& stmt);
  void emitAugAssignment(const torch::jit::AugAssign& stmt);
  void emitRaise(const torch::jit::Raise& stmt);
  void emitAssert(const torch::jit::Assert& stmt);
  void emitReturn(const torch::jit::Return& stmt);
  void emitContinue(const torch::jit::Continue& stmt);
  void emitBreak(const torch::jit::Break& stmt);
  void emitClosure(const torch::jit::Def& stmt);
  void emitDelete(const torch::jit::Delete& smt);
  void emitExpr(const torch::jit::Expr& tree);

  TorchDef ast_;
  PypetScop parsed_data_;
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
#endif
