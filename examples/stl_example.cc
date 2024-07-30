#include <deque>
#include <list>
#include <set>

#include <thread>
#include <unordered_set>

#include <vector>

#include <chrono>

using namespace std::chrono_literals;

template <template <typename, typename...> class Container>
void test_function() {
  Container<int> c;
  for (int i = 0; i < 300; i++) {
    if constexpr (std::is_same_v<Container<int>, std::set<int>> ||
                  std::is_same_v<Container<int>, std::unordered_set<int>>) {
      c.insert(i);
    } else {
      c.push_back(i);
    }
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
}

int main(int argc, char *argv[]) {

  test_function<std::vector>();
  test_function<std::deque>();
  test_function<std::list>();
  test_function<std::unordered_set>();
  test_function<std::set>();

  return 0;
}