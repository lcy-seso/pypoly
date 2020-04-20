#pragma once

#include "pypet/core/pypet.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/frontend/tree_views.h>

#include <vector>

namespace py = pybind11;

namespace pypet {

using TorchDef = torch::jit::Def;
struct PypetScop;

struct ParserImpl {
  explicit ParserImpl(const TorchDef& def)
      : ast(std::move(def)), parsedData(PypetScop()){};
  void dumpAST() const { std::cout << ast; }

 private:
  TorchDef ast;
  PypetScop parsedData;
};

struct ScopParser {
  explicit ScopParser(const TorchDef& def) : pImpl(new ParserImpl(def)){};
  ~ScopParser() = default;

  void dumpAST() const { pImpl->dumpAST(); }

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

}  // namespace pypet
