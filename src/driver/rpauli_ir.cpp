// Writes the random k-local circuit of the engine (--circuit rpauli) as a sectorq.circuit.v1 file, so that the
// paulistrings-rs baseline (jobs/p_ki.py --case) applies the same rotations. Built with the same compiler and the
// same circuit.hpp as the engine, the random generator draws the same qubits, letters and angles.
//
//   rpauli_ir <out.ir.json> <n> <layers> <k> <seed> <amp>      (engine: --n --layers --locality --seed --amp)
//
// The format follows pauliprop-sectorq-impl/src/engine/circuit_ir.hpp (read only, not included).
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../engine/circuit.hpp"

static std::string pauliText(const BV& v, const Layout& L) {
  std::string s(L.n, 'I');
  for (int q = 0; q < L.n; q++) s[q] = "IZXY"[2 * getX(v.data(), q) + getZ(v.data(), L, q)];
  return s;
}

static void writeOps(FILE* f, const std::vector<Op>& ops, const Layout& L) {
  fputs("[", f);
  for (size_t i = 0; i < ops.size(); i++) {
    const Op& o = ops[i];
    if (o.type != OP_ROT) { fprintf(stderr, "rpauli_ir: only rotations expected\n"); std::exit(1); }
    fprintf(f, "%s\n{\"type\":\"rotation\",\"truncate\":%s,\"qubits\":[", i ? "," : "", o.trunc ? "true" : "false");
    for (size_t j = 0; j < o.qubits.size(); j++) fprintf(f, "%s%d", j ? "," : "", o.qubits[j]);
    fprintf(f, "],\"pauli\":\"%s\",\"angle\":%.17g,\"branching\":%s}", pauliText(o.gen, L).c_str(), o.angle,
            o.branches ? "true" : "false");
  }
  fputs("\n]", f);
}

int main(int argc, char** argv) {
  if (argc != 7) { fprintf(stderr, "usage: rpauli_ir <out> <n> <layers> <k> <seed> <amp>\n"); return 2; }
  const int n = atoi(argv[2]), layers = atoi(argv[3]), k = atoi(argv[4]), seed = atoi(argv[5]);
  const double amp = atof(argv[6]);
  // the engine's defaults: observable Z on qubit n/2, truncation every 10 operations (pp_engine.cpp buildProblem)
  const Problem P = makeRandomPauli(n, layers, k, seed, amp, n / 2, 10);
  const std::vector<Op> compiled = compileForEngine(P, true);
  FILE* f = fopen(argv[1], "w");
  if (!f) { perror(argv[1]); return 1; }
  fprintf(f, "{\"schema\":\"sectorq.circuit.v1\",\"qubits\":%d,\"qubit_order\":\"q0_left\",\"rotation_convention\":"
             "\"U=exp(-i*angle*P/2); observable update U_dagger*O*U\",\"precision\":\"float64\",\"initial_state\":\"zero\","
             "\"frame\":true,\"truncation\":{\"kind\":\"absolute\",\"cutoff\":0,\"weight\":0,\"clifford_rotation_dust\":1e-13},"
             "\"observable\":[", P.L.n);
  for (size_t i = 0; i < P.observable.size(); i++)
    fprintf(f, "%s{\"pauli\":\"%s\",\"coefficient\":%.17g}", i ? "," : "", pauliText(P.observable[i].first, P.L).c_str(),
            P.observable[i].second);
  fputs("],\"constructed_state_order\":", f);
  writeOps(f, P.ops, P.L);
  fputs(",\"processing_order\":", f);
  writeOps(f, compiled, P.L);
  fputs("}\n", f);
  return fclose(f) == 0 ? 0 : 1;
}
