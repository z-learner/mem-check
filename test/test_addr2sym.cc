#include "addr2sym.hpp"
#include <gtest/gtest.h>
namespace test {

int func(int a, int b) {
  int c = a + b;
  return c;
}

} // namespace test

TEST(Addr2SymTest, Test) {
  void *func_ptr = reinterpret_cast<void *>(test::func);
  auto name = addr2sym(func_ptr);
  EXPECT_EQ(name, "test::func(int, int)");
}