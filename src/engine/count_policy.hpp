#pragma once
#include <cstdint>

// Experimental controls. A zero fixed count preserves adaptive behavior.
// The bootstrap threshold is the number of GPU workers, rounded up to a power of two.
struct CountPolicy {
  int fixedRows = 0, activationRows = 0;
  bool enabled = false, active = false;
  CountPolicy(int fixedCount, int workers) : enabled(fixedCount != 0) {
    while ((1 << fixedRows) < fixedCount) ++fixedRows;
    while ((1 << activationRows) < workers) ++activationRows;
  }
  int select(int adaptiveRows) {
    if (enabled && adaptiveRows >= activationRows) active = true;
    return active ? fixedRows : adaptiveRows;
  }
};

inline uint64_t policyMix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Candidate index, window start and spare-row domain are independent. Generating
// extra rows/candidates in one window cannot consume another window's RNG stream.
inline uint64_t policyStreamSeed(uint64_t seed, uint64_t windowStart, uint64_t stream) {
  return policyMix(seed ^ policyMix(windowStart) ^ policyMix(stream + 0x100000000ULL));
}
