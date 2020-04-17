#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace py = pybind11;

namespace pypet {

struct ScopParser {
  explicit ScopParser(const std::string& info) : ast_(info) {}

 private:
  std::string ast_;
};

}  // namespace pypet
