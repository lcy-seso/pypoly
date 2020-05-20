#include "c10/util/Logging.h"
#include "gtest/gtest.h"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  c10::ShowLogInfoToStderr();
  return RUN_ALL_TESTS();
}
