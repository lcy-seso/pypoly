#pragma once

#include <c10/util/SmallVector.h>
#include <c10/util/intrusive_ptr.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/frontend/tree_views.h>

#include <vector>

namespace py = pybind11;

namespace pypet {

using TorchDef = torch::jit::Def;

struct ScopParser {
  explicit ScopParser(const TorchDef& def) : ast_(std::move(def)) {}

  void dump() const { std::cout << ast_; }

 private:
  TorchDef ast_;
};
}  // namespace pypet
