// CPU integration test of the actual engine's window/row selector. This does
// not initialize MPI, CUDA workers or a cuPauliProp context, and is not timing data.
#define main sectorq_engine_main
#include "../src/engine/pp_engine.cpp"
#undef main
#include <cassert>

static long sampledMaximum(const Window& w, const std::vector<uint64_t>& sample) {
  std::vector<long> counts(1 << w.g, 0);
  for (size_t i = 0; i < sample.size(); i += g_L.words())
    counts[ownerCpu(sample.data() + i, w.rows, w.g)]++;
  return *std::max_element(counts.begin(), counts.end());
}

static void checkConservation(const Window& w) {
  std::vector<BV> masks = w.spare;
  for (int j = 0; j < w.g; ++j)
    masks.emplace_back(w.rows.begin() + j * g_L.words(), w.rows.begin() + (j + 1) * g_L.words());
  for (int k = w.b; k < w.e; ++k) {
    const Op& op = g_ops[k];
    for (auto& mask : masks) {
      if (op.type == OP_ROT && op.branches)
        assert(dotParity(mask.data(), op.gen.data(), g_L.words()) == 0);
      if (changesBits(op)) mapMask(op, mask.data(), g_L);
    }
  }
}

int main() {
  int checks = 0;
  for (std::string name : {"ki_lab", "tfim_hh", "ki_noisy", "tfim_sq", "xxz", "qaoa", "hea"}) {
    Opts o; o.circuit = name; o.policySeed = 7;
    if (name == "tfim_hh" || name == "tfim_sq") { o.theta = 0.2; o.phi = -0.3; }
    if (name == "tfim_sq") { o.lx = o.ly = 11; o.steps = 10; }
    if (name == "xxz") { o.lx = o.ly = 8; o.steps = 6; o.jz = 0.15; }
    if (name == "hea") { o.n = 100; o.layers = 16; o.amp = 0.4; }
    if (name == "qaoa") { o.n = 100; o.layers = 4; }
    Problem p = buildProblem(o);
    g_L = p.L;
    g_ops = compileForEngine(p, true);
    std::vector<uint64_t> sample;
    std::mt19937_64 input(19);
    for (int i = 0; i < 2048; ++i) {
      BV value(g_L.words(), 0);
      for (int b = 0; b < 2 * g_L.n; ++b)
        if ((input() % 11) == 0) flipBit(value.data(), g_L, b);
      sample.insert(sample.end(), value.begin(), value.end());
    }
    for (int start : {0, (int)g_ops.size() / 3, 2 * (int)g_ops.size() / 3}) {
      for (int rows = 4; rows <= 8; ++rows) {
        std::mt19937_64 legacy1(7), legacy2(999);
        Opts one = o; one.cands = 1;
        Window first = planWindow(start, rows, sample, one, legacy1);
        Window eight = planWindow(start, rows, sample, o, legacy1);
        Window repeated = planWindow(start, rows, sample, o, legacy2);
        assert(first.b == eight.b && first.e == eight.e && first.g == eight.g);
        assert(sampledMaximum(eight, sample) <= sampledMaximum(first, sample));
        assert(eight.rows == repeated.rows && eight.spare == repeated.spare);
        assert(first.spare == eight.spare);
        checkConservation(first);
        checkConservation(eight);
        // Neither the candidate count nor a different row count can consume the
        // random stream used for the next window or for overflow spare rows.
        planWindow(start, rows - 1, sample, one, legacy1);
        Window after = planWindow(start, rows, sample, o, legacy1);
        assert(after.rows == eight.rows && after.spare == eight.spare);
        ++checks;
      }
    }
    printf("%s: valid rows/spares, common candidate prefix, independent window streams passed\n", name.c_str());
  }
  printf("%d actual-window selection cases passed; no GPU execution performed\n", checks);
}
