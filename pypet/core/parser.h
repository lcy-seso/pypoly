#pragma once

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
  TorchParser(std::string src) : src_(std::move(src)) {}

  TorchDef Parse() {
    torch::jit::Parser p(std::make_shared<torch::jit::Source>(src_));
    auto ast = TorchDef(p.parseFunction(/*is_method=*/true));
    return ast;
  }

  std::string src_;
};

struct ParserImpl {
  explicit ParserImpl(const TorchDef& def)
      : ast(std::move(def)), parsed_data(PypetScop()){};
  void dumpAST() const { std::cout << ast; }

 private:
  TorchDef ast;
  PypetScop parsed_data;
};

struct ScopParser {
  explicit ScopParser(const TorchDef& def) : pImpl(new ParserImpl(def)) {}
  ~ScopParser() = default;

  void DumpAST() const { pImpl->dumpAST(); }

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

}  // namespace pypet
