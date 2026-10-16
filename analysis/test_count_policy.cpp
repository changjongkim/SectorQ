#include "../src/engine/count_policy.hpp"
#include <cassert>
#include <cstdio>
#include <random>
#include <set>

int main() {
  CountPolicy adaptive(0, 16), fixed(64, 16);
  for (int g : {0, 1, 2, 3}) {
    assert(adaptive.select(g) == g && !adaptive.active);
    assert(fixed.select(g) == g && !fixed.active);
  }
  assert(fixed.select(4) == 6 && fixed.active);
  for (int g : {8, 7, 3, 1, 0}) {
    assert(fixed.select(g) == 6 && fixed.active);
    assert(adaptive.select(g) == g && !adaptive.active);
  }
  std::set<uint64_t> domains;
  for (int window : {0, 12, 73}) {
    for (uint64_t c : {0ULL, 1ULL, 7ULL, 0x100000000ULL}) {
      auto seed = policyStreamSeed(7, window, c);
      assert(domains.insert(seed).second);
      std::mt19937_64 shortRows(seed), longRows(seed);
      for (int k = 0; k < 4; ++k) assert(shortRows() == longRows());
      for (int k = 0; k < 100; ++k) longRows();
      std::mt19937_64 next1(policyStreamSeed(7, window + 1, c));
      std::mt19937_64 next2(policyStreamSeed(7, window + 1, c));
      assert(next1() == next2());
    }
  }
  assert(policyStreamSeed(7, 12, 0) != policyStreamSeed(11, 12, 0));
  puts("count activation, persistence, adaptive fallback and stream independence passed");
}
