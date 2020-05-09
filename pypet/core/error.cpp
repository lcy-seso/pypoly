#include "pypet/core/error.h"

namespace pypet {

Error::Error(const Error& e)
    : ss(e.ss.str()), context(e.context), the_message(e.the_message) {}

Error::Error(torch::jit::SourceRange r) : context(std::move(r)) {}

const char* Error::what() const noexcept {
  std::stringstream msg;
  msg << "\n" << ss.str();
  msg << ":\n";
  context.highlight(msg);

  the_message = msg.str();
  return the_message.c_str();
}
}  // namespace pypet
