#ifndef _SUGARED_VALUE_H
#define _SUGARED_VALUE_H

#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/sugared_value.h>

namespace pypet {

struct SugaredValue;

using SugaredValuePtr = std::shared_ptr<SugaredValue>;

// TorchScript AST cantains sugared values. For example, loop iterator is a node
// with kind 'Apply' which may call python function range, zip, etc.; As in
// PyTorch JIT parser, sugared value is to temporarily hold these sugared value.
struct SugaredValue : public std::enable_shared_from_this<SugaredValue> {
  // TODO(Ying): Interfaces for SugaredValue is not fully designed. Current
  // codes serves as a placeholder.
  virtual std::string kind() const = 0;

  virtual std::shared_ptr<SugaredValue> call(
      const torch::jit::SourceRange& loc) {
    throw torch::jit::ErrorReport(loc) << "cannot call a " << kind();
  }

  // This function is called when to convert a SugaredValue to its iterator.
  // For example, when iterating through a Dict we iterate over its keys
  // Extract information of iteratioin domain by implementing this function.
  virtual std::shared_ptr<SugaredValue> iter(
      const torch::jit::SourceRange& loc) {
    throw torch::jit::ErrorReport(loc)
        << kind() << " cannot be used as an iterable";
  }

  // expression for ith elemement for iterable value
  virtual std::shared_ptr<SugaredValue> getitem(
      const torch::jit::SourceRange& loc) {
    throw torch::jit::ErrorReport(loc) << "'" << kind() << "'"
                                       << " object is not subscriptable";
  }

  virtual ~SugaredValue() = default;
};
}  // namespace pypet
#endif
