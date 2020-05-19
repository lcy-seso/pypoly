#ifndef UTIL_H_
#define UTIL_H_

#include "c10/util/Logging.h"

namespace pypoly {
namespace pypet {

#define UNIMPLEMENTED() LOG(FATAL) << "UNIMPLEMENTED"

}  // namespace pypet
}  // namespace pypoly

#endif  // UTIL_H_
