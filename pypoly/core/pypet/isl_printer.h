#ifndef PYPOLY_CORE_PYPET_ISL_PRINTER_H_
#define PYPOLY_CORE_PYPET_ISL_PRINTER_H_

#include "util.h"

namespace pypoly {
namespace pypet {

inline void isl_printer_to_stdout(struct isl_printer* p) {
  char* str = isl_printer_get_str(p);
  isl_printer_free(p);
  std::cout << std::string(str) << std::endl;
  free(str);
}

inline void print_isl_multi_pw_aff(isl_multi_pw_aff* pw_multi_aff) {
  isl_printer* p = isl_printer_to_str(isl_multi_pw_aff_get_ctx(pw_multi_aff));
  p = isl_printer_print_multi_pw_aff(p, pw_multi_aff);
  isl_printer_to_stdout(p);
}

inline void print_isl_pw_aff(isl_pw_aff* pw_aff) {
  isl_printer* p = isl_printer_to_str(isl_pw_aff_get_ctx(pw_aff));
  p = isl_printer_print_pw_aff(p, pw_aff);
  isl_printer_to_stdout(p);
}

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_ISL_PRINTER_H_
