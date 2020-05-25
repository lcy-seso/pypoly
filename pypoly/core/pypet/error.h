#ifndef PYPOLY_CORE_PYPET_ERROR_H_
#define PYPOLY_CORE_PYPET_ERROR_H_

#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/tree.h>

namespace pypoly {
namespace pypet {

struct Error : public std::exception {
  Error(const Error& e);
  explicit Error(torch::jit::SourceRange r);
  explicit Error(const torch::jit::TreeRef& tree) : Error(tree->range()) {}

  const char* what() const noexcept override;

 private:
  template <typename T>
  friend const Error& operator<<(const Error& e, const T& t) {
    e.ss << t;
    return e;
  };

  mutable std::stringstream ss;
  torch::jit::SourceRange context;
  mutable std::string the_message;
};

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_ERROR_H_
