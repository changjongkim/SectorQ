// Circuit description for the engine: operations, their action on Pauli bits, and the
// evaluation circuits.
#pragma once

#include <cupauliprop.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gf2.hpp"
#include "../driver/ibm_eagle.hpp"

enum OpType { OP_ROT, OP_CLIFF, OP_NOISE };

struct Op {
  OpType type = OP_ROT;
  std::vector<int32_t> qubits;
  std::vector<cupaulipropPauliKind_t> paulis;   // OP_ROT: one Pauli per qubit
  double angle = 0;                             // OP_ROT: exp(-i angle/2 P)
  cupaulipropCliffordGateKind_t cliff = CUPAULIPROP_CLIFFORD_GATE_I;
  double prob[4] = {1, 0, 0, 0};                // OP_NOISE: [p_I, p_X, p_Y, p_Z]
  bool trunc = false;                           // truncate after this operation
  // derived by finalize()
  BV gen;                                       // OP_ROT: bits of P
  bool branches = false;                        // OP_ROT with a non-Clifford angle
  bool flips = false;                           // OP_ROT with angle = odd multiple of pi/2
};

struct Problem {
  std::string name;
  Layout L;
  std::vector<Op> ops;                                  // order of application to the state
  std::vector<std::pair<BV, double>> observable;        // sum of Pauli strings
};

static const double PI_ = 3.14159265358979323846;

// ---------------------------------------------------------------- bit maps
// How an operation changes the bits of a Pauli string in the Heisenberg picture. Every map
// used here is its own inverse. Rotations with a non-Clifford angle are not maps (they branch).

inline bool changesBits(const Op& o) {
  if (o.type == OP_NOISE) return false;
  if (o.type == OP_ROT) return o.flips;
  switch (o.cliff) {
    case CUPAULIPROP_CLIFFORD_GATE_I: case CUPAULIPROP_CLIFFORD_GATE_X:
    case CUPAULIPROP_CLIFFORD_GATE_Y: case CUPAULIPROP_CLIFFORD_GATE_Z: return false;
    default: return true;
  }
}

// v <- T(v): the string after the operation.
inline void mapVec(const Op& o, uint64_t* v, const Layout& L) {
  if (!changesBits(o)) return;
  if (o.type == OP_ROT) {
    if (symplectic(v, o.gen.data(), L)) for (int w = 0; w < L.words(); w++) v[w] ^= o.gen[w];
    return;
  }
  const int a = o.qubits[0];
  switch (o.cliff) {
    case CUPAULIPROP_CLIFFORD_GATE_H: case CUPAULIPROP_CLIFFORD_GATE_SQRTY: {
      const bool x = getX(v, a), z = getZ(v, L, a);
      if (x != z) { flipX(v, a); flipZ(v, L, a); }
      break; }
    case CUPAULIPROP_CLIFFORD_GATE_S: case CUPAULIPROP_CLIFFORD_GATE_SQRTZ:
      if (getX(v, a)) flipZ(v, L, a);
      break;
    case CUPAULIPROP_CLIFFORD_GATE_SQRTX:
      if (getZ(v, L, a)) flipX(v, a);
      break;
    case CUPAULIPROP_CLIFFORD_GATE_CZ: {
      const int b = o.qubits[1];
      const bool xa = getX(v, a), xb = getX(v, b);
      if (xb) flipZ(v, L, a);
      if (xa) flipZ(v, L, b);
      break; }
    default:
      fprintf(stderr, "Clifford gate kind %d has no bit map in the engine\n", (int)o.cliff);
      std::exit(1);
  }
}

// m <- T^T(m): the row that gives, on strings after the operation, the value the row m gave before.
inline void mapMask(const Op& o, uint64_t* m, const Layout& L) {
  if (!changesBits(o)) return;
  if (o.type == OP_ROT) {
    if (dotParity(m, o.gen.data(), L.words()))
      for (int w = 0; w < L.nw; w++) { m[w] ^= o.gen[L.nw + w]; m[L.nw + w] ^= o.gen[w]; }
    return;
  }
  const int a = o.qubits[0];
  switch (o.cliff) {
    case CUPAULIPROP_CLIFFORD_GATE_H: case CUPAULIPROP_CLIFFORD_GATE_SQRTY: {
      const bool x = getX(m, a), z = getZ(m, L, a);
      if (x != z) { flipX(m, a); flipZ(m, L, a); }
      break; }
    case CUPAULIPROP_CLIFFORD_GATE_S: case CUPAULIPROP_CLIFFORD_GATE_SQRTZ:
      if (getZ(m, L, a)) flipX(m, a);
      break;
    case CUPAULIPROP_CLIFFORD_GATE_SQRTX:
      if (getX(m, a)) flipZ(m, L, a);
      break;
    case CUPAULIPROP_CLIFFORD_GATE_CZ: {
      const int b = o.qubits[1];
      const bool za = getZ(m, L, a), zb = getZ(m, L, b);
      if (za) flipX(m, b);
      if (zb) flipX(m, a);
      break; }
    default:
      fprintf(stderr, "Clifford gate kind %d has no bit map in the engine\n", (int)o.cliff);
      std::exit(1);
  }
}

// ---------------------------------------------------------------- construction helpers

inline Op rot(std::vector<int32_t> q, std::vector<cupaulipropPauliKind_t> p, double angle) {
  Op o; o.type = OP_ROT; o.qubits = std::move(q); o.paulis = std::move(p); o.angle = angle; return o;
}
inline Op cliff(cupaulipropCliffordGateKind_t k, std::vector<int32_t> q) {
  Op o; o.type = OP_CLIFF; o.cliff = k; o.qubits = std::move(q); return o;
}
inline Op depol(int32_t q, double p) {
  Op o; o.type = OP_NOISE; o.qubits = {q}; o.prob[0] = 1 - p; o.prob[1] = o.prob[2] = o.prob[3] = p / 3; return o;
}

inline BV pauliBits(const Layout& L, const std::vector<int32_t>& q, const std::vector<cupaulipropPauliKind_t>& p) {
  BV v(L.words(), 0);
  for (size_t i = 0; i < q.size(); i++) {
    if (p[i] == CUPAULIPROP_PAULI_X || p[i] == CUPAULIPROP_PAULI_Y) flipX(v.data(), q[i]);
    if (p[i] == CUPAULIPROP_PAULI_Z || p[i] == CUPAULIPROP_PAULI_Y) flipZ(v.data(), L, q[i]);
  }
  return v;
}

// Fills the derived fields; truncEvery > 0 sets a truncation after every truncEvery-th operation.
inline void finalize(Problem& P, int truncEvery) {
  for (size_t i = 0; i < P.ops.size(); i++) {
    Op& o = P.ops[i];
    if (truncEvery > 0) o.trunc = (i % truncEvery == 0);
    if (o.type != OP_ROT) continue;
    o.gen = pauliBits(P.L, o.qubits, o.paulis);
    const double k = o.angle / (PI_ / 2);
    const double kr = std::round(k);
    if (std::fabs(k - kr) < 1e-12) { o.branches = false; o.flips = (((long)kr) % 2 != 0); }
    else { o.branches = true; o.flips = false; }
  }
}

inline Layout makeLayout(int n) {
  Layout L; L.n = n;
  int32_t nw = 0;
  cupaulipropGetNumPackedIntegers(n, &nw);
  L.nw = nw;
  return L;
}

// ---------------------------------------------------------------- lattices

typedef std::vector<std::pair<int, int>> Edges;

inline Edges heavyHexEagle() {
  Edges e;
  const int32_t (*layers[3])[2] = {ZZ_QUBITS_RED, ZZ_QUBITS_BLUE, ZZ_QUBITS_GREEN};
  for (int l = 0; l < 3; l++)
    for (int i = 0; i < NUM_ROTATIONS_PER_LAYER; i++) e.push_back({layers[l][i][0], layers[l][i][1]});
  return e;
}
inline Edges squareLattice(int lx, int ly) {
  Edges e;
  for (int y = 0; y < ly; y++) for (int x = 0; x + 1 < lx; x++) e.push_back({y * lx + x, y * lx + x + 1});
  for (int y = 0; y + 1 < ly; y++) for (int x = 0; x < lx; x++) e.push_back({y * lx + x, (y + 1) * lx + x});
  return e;
}
inline Edges lineLattice(int n) {
  Edges e;
  for (int i = 0; i + 1 < n; i += 2) e.push_back({i, i + 1});
  for (int i = 1; i + 1 < n; i += 2) e.push_back({i, i + 1});
  return e;
}
inline Edges random3Regular(int n, int seed) {
  std::mt19937_64 rng(seed);
  for (int attempt = 0; attempt < 10000; attempt++) {
    std::vector<int> stubs;
    for (int i = 0; i < n; i++) for (int k = 0; k < 3; k++) stubs.push_back(i);
    std::shuffle(stubs.begin(), stubs.end(), rng);
    Edges e; bool ok = true;
    for (size_t i = 0; i + 1 < stubs.size() && ok; i += 2) {
      int a = std::min(stubs[i], stubs[i + 1]), b = std::max(stubs[i], stubs[i + 1]);
      if (a == b) ok = false;
      for (auto& x : e) if (x.first == a && x.second == b) ok = false;
      e.push_back({a, b});
    }
    if (ok) return e;
  }
  fprintf(stderr, "could not build a 3-regular graph\n");
  std::exit(1);
}
// Two-colouring of a bipartite lattice (breadth first).
inline std::vector<int> twoColour(int n, const Edges& e) {
  std::vector<std::vector<int>> adj(n);
  for (auto& x : e) { adj[x.first].push_back(x.second); adj[x.second].push_back(x.first); }
  std::vector<int> c(n, -1);
  for (int s = 0; s < n; s++) {
    if (c[s] >= 0) continue;
    c[s] = 0;
    std::vector<int> q = {s};
    for (size_t h = 0; h < q.size(); h++)
      for (int v : adj[q[h]]) if (c[v] < 0) { c[v] = 1 - c[q[h]]; q.push_back(v); }
  }
  return c;
}

inline void singleZ(Problem& P, int q) {
  P.observable.push_back({pauliBits(P.L, {q}, {CUPAULIPROP_PAULI_Z}), 1.0});
}

// ---------------------------------------------------------------- the evaluation circuits

// 1. Kicked Ising on the IBM Eagle lattice in Clifford-frame form: the ZZ(pi/2) layers are absorbed
//    into the rotation generators, so only 127 rotations per step touch the data.
inline Problem makeKickedIsingFrame(int steps, double thetaOverPi, int obsQ, int truncEvery) {
  Problem P; P.name = "kicked Ising, Eagle heavy-hex, Clifford frame"; P.L = makeLayout(NUM_CIRCUIT_QUBITS);
  Edges e = heavyHexEagle();
  std::vector<std::vector<int32_t>> nb(P.L.n);
  for (auto& x : e) { nb[x.first].push_back(x.second); nb[x.second].push_back(x.first); }
  for (int n = 0; n < steps; n++)
    for (int32_t q = 0; q < P.L.n; q++) {
      const int deg = (int)nb[q].size();
      double sign = ((deg % 2) && ((n / 2) % 2)) ? -1.0 : 1.0;
      std::vector<int32_t> qs = {q};
      std::vector<cupaulipropPauliKind_t> ps;
      if (n % 2 == 0) {
        ps.push_back(CUPAULIPROP_PAULI_X);
      } else {
        ps.push_back(deg % 2 ? CUPAULIPROP_PAULI_Y : CUPAULIPROP_PAULI_X);
        for (int32_t b : nb[q]) { qs.push_back(b); ps.push_back(CUPAULIPROP_PAULI_Z); }
        if (deg % 4 == 2 || deg % 4 == 3) sign *= -1.0;
      }
      Op o = rot(qs, ps, sign * thetaOverPi * PI_);
      o.trunc = ((271 * n + q) % truncEvery == 0) || (q == 0 && n > 0);   // same points as the original circuit
      P.ops.push_back(o);
    }
  singleZ(P, obsQ);
  finalize(P, 0);
  return P;
}

// 1/2/3/7. Trotterised Ising: X rotations on every qubit, ZZ rotations on every edge, optional
//    depolarising noise on every qubit after each step. zz = -pi/2 gives the kicked Ising circuit.
inline Problem makeIsing(const std::string& name, int n, const Edges& e, int steps, double thetaOverPi,
                         double zzOverPi, double noiseP, int obsQ, int truncEvery) {
  Problem P; P.name = name; P.L = makeLayout(n);
  for (int s = 0; s < steps; s++) {
    for (int32_t q = 0; q < n; q++) P.ops.push_back(rot({q}, {CUPAULIPROP_PAULI_X}, thetaOverPi * PI_));
    for (auto& x : e) P.ops.push_back(rot({x.first, x.second}, {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z}, zzOverPi * PI_));
    if (noiseP > 0) for (int32_t q = 0; q < n; q++) P.ops.push_back(depol(q, noiseP));
  }
  singleZ(P, obsQ);
  finalize(P, truncEvery);
  return P;
}

// 4. XXZ Heisenberg model from the Neel state: XX, YY, ZZ rotations on every edge.
inline Problem makeXXZ(const std::string& name, int n, const Edges& e, int steps, double jxyOverPi,
                       double jzOverPi, int obsQ, int truncEvery) {
  Problem P; P.name = name; P.L = makeLayout(n);
  std::vector<int> col = twoColour(n, e);
  for (int32_t q = 0; q < n; q++) if (col[q]) P.ops.push_back(cliff(CUPAULIPROP_CLIFFORD_GATE_X, {q}));
  for (int s = 0; s < steps; s++)
    for (auto& x : e) {
      P.ops.push_back(rot({x.first, x.second}, {CUPAULIPROP_PAULI_X, CUPAULIPROP_PAULI_X}, jxyOverPi * PI_));
      P.ops.push_back(rot({x.first, x.second}, {CUPAULIPROP_PAULI_Y, CUPAULIPROP_PAULI_Y}, jxyOverPi * PI_));
      P.ops.push_back(rot({x.first, x.second}, {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z}, jzOverPi * PI_));
    }
  singleZ(P, obsQ);
  finalize(P, truncEvery);
  return P;
}

// 5. QAOA for MaxCut on a random 3-regular graph; the observable is the sum of Z_a Z_b over edges.
inline Problem makeQaoa(int n, int p, int seed, int truncEvery) {
  Problem P; P.name = "QAOA MaxCut, random 3-regular graph"; P.L = makeLayout(n);
  Edges e = random3Regular(n, seed);
  for (int32_t q = 0; q < n; q++) P.ops.push_back(cliff(CUPAULIPROP_CLIFFORD_GATE_H, {q}));
  for (int l = 0; l < p; l++) {
    const double gamma = 0.25 * PI_ * (l + 1) / p, beta = 0.20 * PI_ * (1.0 - (double)l / p);
    for (auto& x : e) P.ops.push_back(rot({x.first, x.second}, {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z}, gamma));
    for (int32_t q = 0; q < n; q++) P.ops.push_back(rot({q}, {CUPAULIPROP_PAULI_X}, 2 * beta));
  }
  for (auto& x : e)
    P.observable.push_back({pauliBits(P.L, {x.first, x.second}, {CUPAULIPROP_PAULI_Z, CUPAULIPROP_PAULI_Z}), 1.0});
  finalize(P, truncEvery);
  return P;
}

// 6. Hardware-efficient ansatz on a line: RY and RZ with random angles on every qubit, then CZ on
//    alternating bonds.
inline Problem makeHea(int n, int layers, int seed, double ampOverPi, int obsQ, int truncEvery) {
  Problem P; P.name = "hardware-efficient ansatz, line"; P.L = makeLayout(n);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-ampOverPi * PI_, ampOverPi * PI_);
  Edges e = lineLattice(n);
  for (int l = 0; l < layers; l++) {
    for (int32_t q = 0; q < n; q++) P.ops.push_back(rot({q}, {CUPAULIPROP_PAULI_Y}, u(rng)));
    for (int32_t q = 0; q < n; q++) P.ops.push_back(rot({q}, {CUPAULIPROP_PAULI_Z}, u(rng)));
    for (auto& x : e) P.ops.push_back(cliff(CUPAULIPROP_CLIFFORD_GATE_CZ, {x.first, x.second}));
  }
  singleZ(P, obsQ);
  finalize(P, truncEvery);
  return P;
}

// Random k-local Pauli rotations (experiment E5 of the 2026-11-12 review plan): each layer rotates n generators, each on k
// distinct qubits drawn uniformly from all n, with letters X, Y or Z drawn uniformly, by an angle uniform in
// [-amp*pi, amp*pi]. No Clifford gates. A larger k makes the generators more diverse, so that a window reaches the rank
// limit after fewer rotations.
inline Problem makeRandomPauli(int n, int layers, int k, int seed, double ampOverPi, int obsQ, int truncEvery) {
  Problem P; P.name = "random " + std::to_string(k) + "-local Pauli rotations"; P.L = makeLayout(n);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-ampOverPi * PI_, ampOverPi * PI_);
  std::uniform_int_distribution<int> letter(0, 2);
  const cupaulipropPauliKind_t kinds[3] = {CUPAULIPROP_PAULI_X, CUPAULIPROP_PAULI_Y, CUPAULIPROP_PAULI_Z};
  std::vector<int32_t> all(n);
  for (int q = 0; q < n; q++) all[q] = q;
  for (int l = 0; l < layers; l++)
    for (int r = 0; r < n; r++) {
      std::shuffle(all.begin(), all.end(), rng);
      std::vector<int32_t> q(all.begin(), all.begin() + k);
      std::sort(q.begin(), q.end());
      std::vector<cupaulipropPauliKind_t> p(k);
      for (auto& x : p) x = kinds[letter(rng)];
      P.ops.push_back(rot(q, p, u(rng)));
    }
  singleZ(P, obsQ);
  finalize(P, truncEvery);
  return P;
}

// ---------------------------------------------------------------- Clifford frame
// A Hermitian Pauli string with a sign.
struct SignedPauli { BV bits; int sign = 1; };

// i-exponent picked up when multiplying single-qubit Paulis (x1,z1) * (x2,z2) (Aaronson-Gottesman g).
inline int phaseG(int x1, int z1, int x2, int z2) {
  if (!x1 && !z1) return 0;
  if (x1 && z1) return z2 - x2;
  if (x1 && !z1) return z2 * (2 * x2 - 1);
  return x2 * (1 - 2 * z2);
}

// P <- G P G^dagger for a Clifford operation G (a Clifford gate or a rotation by a multiple of pi/2);
// with inverse, P <- G^dagger P G. Returns false if the gate kind is not supported in the frame.
inline bool conjugate(const Op& g, SignedPauli& P, const Layout& L, bool inverse = false) {
  uint64_t* v = P.bits.data();
  if (g.type == OP_ROT) {
    long k = ((long)std::llround(g.angle / (PI_ / 2)) % 4 + 4) % 4;
    if (inverse) k = (4 - k) % 4;
    if (k == 0 || !symplectic(v, g.gen.data(), L)) return true;
    if (k == 2) { P.sign = -P.sign; return true; }
    int G = 0;                                     // R * P = i^G * (letters of the XOR)
    for (int32_t q : g.qubits)
      G += phaseG(getX(g.gen.data(), q), getZ(g.gen.data(), L, q), getX(v, q), getZ(v, L, q));
    G = ((G % 4) + 4) % 4;                         // 1 or 3 for anticommuting strings
    for (int w = 0; w < L.words(); w++) v[w] ^= g.gen[w];
    // k == 1: -i R P = i^(G-1) ; k == 3: +i R P = i^(G+1)
    const int e = (k == 1) ? (G + 3) % 4 : (G + 1) % 4;
    if (e == 2) P.sign = -P.sign;
    return true;
  }
  if (g.type != OP_CLIFF) return false;
  const int a = g.qubits[0];
  const bool x = getX(v, a), z = getZ(v, L, a);
  switch (g.cliff) {
    case CUPAULIPROP_CLIFFORD_GATE_I: return true;
    case CUPAULIPROP_CLIFFORD_GATE_X: if (z) P.sign = -P.sign; return true;
    case CUPAULIPROP_CLIFFORD_GATE_Z: if (x) P.sign = -P.sign; return true;
    case CUPAULIPROP_CLIFFORD_GATE_Y: if (x != z) P.sign = -P.sign; return true;
    case CUPAULIPROP_CLIFFORD_GATE_H:
      if (x && z) P.sign = -P.sign;
      if (x != z) { flipX(v, a); flipZ(v, L, a); }
      return true;
    case CUPAULIPROP_CLIFFORD_GATE_S:
      if (inverse ? (x && !z) : (x && z)) P.sign = -P.sign;   // S: X->Y, Y->-X ; S^dagger: X->-Y, Y->X
      if (x) flipZ(v, L, a);
      return true;
    case CUPAULIPROP_CLIFFORD_GATE_CZ: {
      const int b = g.qubits[1];
      const bool xb = getX(v, b), zb = getZ(v, L, b);
      if (x && xb && (z != zb)) P.sign = -P.sign;
      if (xb) flipZ(v, L, a);
      if (x) flipZ(v, L, b);
      return true; }
    default: return false;
  }
}

inline void bitsToLetters(const BV& bits, const Layout& L, std::vector<int32_t>& q, std::vector<cupaulipropPauliKind_t>& p) {
  q.clear(); p.clear();
  for (int i = 0; i < L.n; i++) {
    const bool x = getX(bits.data(), i), z = getZ(bits.data(), L, i);
    if (!x && !z) continue;
    q.push_back(i);
    p.push_back(x && z ? CUPAULIPROP_PAULI_Y : (x ? CUPAULIPROP_PAULI_X : CUPAULIPROP_PAULI_Z));
  }
}

// Turns the circuit into the list of operations the engine applies to the data, in the order it
// applies them (last gate of the circuit first). With useFrame, Clifford operations are not applied:
// they are collected in a frame A and every later rotation is replaced by the rotation about
// A Q A^dagger. The frame is applied to the data only where something needs the true strings: before
// a noise channel and before the final trace.
inline std::vector<Op> compileForEngine(const Problem& P, bool useFrame, int* deferredOut = nullptr) {
  const Layout& L = P.L;
  std::vector<Op> out;
  std::vector<const Op*> frame;          // deferred Clifford operations, in processing order
  int deferred = 0;
  // Applying the frame to the data means D <- A^dagger D A with A = G_1 ... G_m. If that map leaves
  // every string's bits unchanged it is a Pauli conjugation, i.e. only signs: it is emitted as at most
  // 2n single-qubit Pauli gates instead of the m deferred gates.
  auto flush = [&]() {
    if (frame.empty()) return;
    bool pauliOnly = true;
    std::vector<int> signX(L.n, 1), signZ(L.n, 1);
    for (int q = 0; q < L.n && pauliOnly; q++)
      for (int t = 0; t < 2 && pauliOnly; t++) {
        SignedPauli b; b.bits.assign(L.words(), 0); b.sign = 1;
        if (t == 0) flipX(b.bits.data(), q); else flipZ(b.bits.data(), L, q);
        const BV before = b.bits;
        for (const Op* g : frame) conjugate(*g, b, L, true);
        if (b.bits != before) pauliOnly = false;
        (t == 0 ? signX[q] : signZ[q]) = b.sign;
      }
    if (pauliOnly) {
      for (int32_t q = 0; q < L.n; q++) {
        if (signX[q] < 0) out.push_back(cliff(CUPAULIPROP_CLIFFORD_GATE_Z, {q}));   // Z flips the sign of X and Y
        if (signZ[q] < 0) out.push_back(cliff(CUPAULIPROP_CLIFFORD_GATE_X, {q}));   // X flips the sign of Z and Y
      }
    } else {
      for (const Op* g : frame) { Op c = *g; c.trunc = false; out.push_back(c); }
    }
    frame.clear();
  };
  for (int i = (int)P.ops.size() - 1; i >= 0; --i) {
    const Op& o = P.ops[i];
    const bool isClifford = (o.type == OP_CLIFF) || (o.type == OP_ROT && !o.branches);
    if (isClifford && useFrame) {
      SignedPauli probe; probe.bits.assign(L.words(), 0);
      if (!conjugate(o, probe, L)) { fprintf(stderr, "Clifford gate kind %d is not supported in the frame\n", (int)o.cliff); std::exit(1); }
      frame.push_back(&o);
      deferred++;
      if (o.trunc) {
        // Coefficient truncation does not depend on the frame: the cut moves to the data operation before it.
        // A cut on a Clifford gate before the first data operation still has to act on the observable, so an
        // identity operation carries it.
        if (out.empty()) out.push_back(cliff(CUPAULIPROP_CLIFFORD_GATE_I, {0}));
        out.back().trunc = true;
      }
      continue;
    }
    if (o.type == OP_NOISE) flush();
    if (o.type == OP_ROT && o.branches && !frame.empty()) {
      SignedPauli q; q.bits = o.gen; q.sign = 1;
      for (int j = (int)frame.size() - 1; j >= 0; --j) conjugate(*frame[j], q, L);
      Op r = o;
      bitsToLetters(q.bits, L, r.qubits, r.paulis);
      r.angle = q.sign * o.angle;
      r.gen = q.bits;
      out.push_back(r);
      continue;
    }
    out.push_back(o);
  }
  flush();
  if (deferredOut) *deferredOut = deferred;
  return out;
}
