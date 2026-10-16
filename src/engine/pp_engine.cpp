// Unified engine for partitioned Pauli propagation.
//
// One MPI process per node, one worker thread per GPU. The operator lives in host memory as
// buckets; a bucket is the set of terms that share the values of g parity rows. For the
// current window of operations the rows are chosen so that every branching rotation leaves
// them unchanged, which makes the buckets independent tasks: a worker loads a bucket, applies
// the whole window on its GPU with local deduplication and truncation, partitions the result
// by the next window's rows and appends the pieces to the next pool.
//
//   windows   found at run time. Rotations are added while the rows that survive them (the GF(2)
//             annihilator of their generators) can still split the terms: either a fixed number of
//             free dimensions is kept (--window ann) or the terms must stay spread over the cosets
//             of the generator span, measured on a sample (--window coset)
//   timing    every term is moved at a re-key, so a window ends right after a truncation, where there
//             are fewest terms (--snap). --pilot 1 instead runs part of one bucket ahead through the
//             longest allowed window (buckets are independent miniatures of the same dynamics) and
//             ends the window where that pilot is smallest
//   rows      random elements of that annihilator, the best-balanced set on a sample of terms
//   overflow  a bucket that outgrows its GPU is split by one more row and requeued
//   frame     Clifford operations are deferred and folded into later rotations
//   storage   buckets are lists of fixed-size host chunks that are pinned once and reused, for
//             GPU copies and for messages between nodes
//   nodes     bucket d belongs to node d mod (number of nodes). After a window every node sends the
//             chunks it produced for other nodes' buckets and adopts the arriving ones (--stream 1 does
//             this while the window runs). GPUs never send to each other; workers make no MPI call
//   restart   the pool at a window boundary, with the window's rows, is a checkpoint
//   verify    parities that the whole circuit conserves are checked on every bucket
//   one GPU   buckets only keep the tasks at the target size (no other GPU could take one). While the operator is
//             below that size it is one bucket and stays on the GPU between windows: no partition, no copy to
//             the host. Above it the buckets go through host memory as on several GPUs
//   noise     consecutive noise channels are applied in one library call
//
// --mode ref   same operation list as one bucket on one GPU (plain cuPauliProp)
// --mode cupp  same operation list through cuPauliProp's own distributed mode (one rank per GPU)

#include <cupauliprop.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/time.h>
#include <unistd.h>

#include "circuit.hpp"
#include "count_policy.hpp"

extern "C" {
void pp_owners16(const uint64_t* xz, int64_t n, int words, const uint64_t* masks, int g, uint16_t* owners);
void pp_sample_terms(const uint64_t* xz, int64_t n, int words, int64_t stride, uint64_t* out);
void pp_counts_sorted16(const uint16_t* owners, int64_t n, int R, int64_t* cnt);
int64_t pp_count_ne16(const uint16_t* owners, int64_t n, int expected);
int pp_partition16(const uint64_t* xz, const double* coef, uint16_t* owners, int64_t n, int words,
                 uint64_t* xzOut, double* coefOut, int32_t* idx);
}

static int g_rank = 0, g_R = 1;
[[noreturn]] static void die(const char* what, int line) {
  fprintf(stderr, "[node %d] FATAL: %s (line %d)\n", g_rank, what, line);
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
  std::exit(1);
}
#define CUPP(x) do { cupaulipropStatus_t s_ = (x); if (s_ != CUPAULIPROP_STATUS_SUCCESS) { \
  fprintf(stderr, "[node %d] cuPauliProp status %d\n", g_rank, (int)s_); die(#x, __LINE__); } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "[node %d] CUDA: %s\n", g_rank, cudaGetErrorString(e_)); die(#x, __LINE__); } } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Wall-clock time of day, to line a run up with samples taken outside the process (GPU utilization).
static double epochNow() { timeval tv; gettimeofday(&tv, nullptr); return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec; }

// Counters of the profiling plugin (bin/libcupp_mpi_prof.so), when that is the library's communication plugin:
// calls, seconds and bytes of every kind of MPI call that cuPauliProp's distributed mode makes on this rank.
static const int CUPP_KINDS = 12;
static const char* CUPP_KIND_NAMES[CUPP_KINDS] = {"barrier", "wait", "test", "send", "isend", "recv", "irecv",
                                                  "bcast", "allreduce", "allreduce_min", "minloc", "allgather"};
static const bool CUPP_KIND_COLLECTIVE[CUPP_KINDS] = {true, false, false, false, false, false, false, true, true, true, true, true};
typedef void (*CuppSnapshotFn)(double*);
static CuppSnapshotFn cuppProfiler() {
  const char* lib = getenv("CUPAULIPROP_COMM_LIB");
  if (!lib) return nullptr;
  void* h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
  if (!h) return nullptr;
  typedef int (*KindsFn)(void);
  KindsFn kinds = (KindsFn)dlsym(h, "cuppprof_num_kinds");
  CuppSnapshotFn snap = (CuppSnapshotFn)dlsym(h, "cuppprof_snapshot");
  return (kinds && snap && kinds() == CUPP_KINDS) ? snap : nullptr;
}

struct Opts {
  std::string circuit = "ki";
  std::string mode = "engine";   // engine | ref | cupp
  int gpus = 1;
  int frame = 1;                 // defer Clifford operations
  // circuit parameters
  int steps = 20, lx = 5, ly = 6, n = 0, obsQ = -1, seed = 7, layers = 4;
  int obsAll = 0;                // the observable is the sum of Z over all qubits (total magnetization), not Z on one qubit
  double theta = 0.25, phi = -0.5, jxy = 0.1, jz = 0.15, noise = 0.0, amp = 0.15;
  // truncation
  double coef = 1e-4;
  int weight = 0, truncEvery = 10;
  double junk = 1e-13;           // dropped after a Clifford-angle rotation applied to the data
  // engine
  double vramFrac = 0.9, expShare = 0.2;
  long target = 4000000;         // desired terms per bucket at the start of a window
  int gmax = 8, cands = 8, annMin = -1;
  int fixedSectors = 0;          // experiment: constant requested count after common bootstrap
  long policySeed = -1;          // -1 retains the legacy RNG; otherwise independent per-window streams
  int policyAudit = 0;           // experiment: row hashes/ranks and count decisions, no extra MPI
  std::string window = "ann";    // ann | coset
  double kappa = 4.0;            // coset rule: the largest coset may hold 1/(kappa * buckets) of the sample
  long sample = 200000;
  int snap = 1;                  // end a window right after a truncation, where the operator is locally smallest
  int pilot = 0;                 // run one bucket ahead through the window and end the window where it is smallest
  int pilotRows = 4;             // the pilot runs on 1/2^pilotRows of that bucket (a sub-bucket by further rows)
  int fixedRows = 0;             // 1: the rows chosen when the terms are first divided stay for the rest of the run (ablation)
  int perGpu = 2;                // at least this many buckets per GPU once there are enough terms
  int stream = 0;                // 1: exchange between nodes while the window runs instead of after it
  long minBucket = 200000;       // ... but never buckets smaller than this at a window start
  double growthCap = 8.0;        // upper limit of the growth per window assumed when the bucket count is chosen
  int earlyPerNode = 1;          // in the first windows: this many buckets per node as soon as they would hold minBucketEarly terms (0 = off)
  long minBucketEarly = 2000;
  long minBucketEarlyEnd = 0;    // > 0: the early buckets are given by the terms a bucket is expected to hold at the END of the window instead (at least this many)
  int trace = 0;                 // print one line per window: operations, rows, sectors, terms in and out
  long shareAbove = 0;           // k > 0: a bucket above k terms is halved at a truncation while GPUs of its node wait (off: it did not pay, job 59483080)
  long chunk = 262144;           // terms per host chunk
  int pinned = 1, verify = 1;
  int overlapPlan = 1;           // several GPUs: choose the rows of the next window while this window runs
  double longWindow = 0.5;       // perGpu buckets per GPU only when the last window ran at least this long (s); else one
  std::string ckptFile, resume;
  int ckptWindow = -1, stopAtCkpt = 0;
  std::string profile;           // write the number of terms before every operation to this file
  int deferDedup = 0;            // k > 0: inside a bucket, merge equal strings only every k operations and before each truncation
  int cuppTuned = 0;             // --mode cupp: apply operations rank-locally and deduplicate globally only before truncations
  int cuppDedupEvery = 0;        // --mode cupp, tuned: also deduplicate globally after this many operations (0 = only before truncations)
};

static Layout g_L;
static std::vector<Op> g_ops;       // operations in processing order
static const int kNoiseGroup = 10;  // at most this many consecutive noise channels per library call

// ---------------------------------------------------------------- host storage
// Buckets are lists of fixed-size chunks. Chunks are pinned once and recycled, so copies to and
// from the GPUs and messages between nodes always use memory that is already locked.

struct Chunk { uint64_t* xz = nullptr; double* cf = nullptr; long n = 0; std::atomic<int>* writers = nullptr; };
struct Seg { uint64_t* xz; double* cf; long n; std::atomic<int>* writers; };   // writers: copies into the chunk still in flight

struct ChunkPool {
  std::mutex m;
  std::vector<Chunk> freeList;
  long cap = 0, allocated = 0;
  std::atomic<long long> allocNs{0};   // time spent allocating (pinning) new chunks, summed over threads
  bool pinned = true;
  long total() { std::lock_guard<std::mutex> lk(m); return allocated; }
  Chunk get() {
    {
      std::lock_guard<std::mutex> lk(m);
      if (!freeList.empty()) { Chunk c = freeList.back(); freeList.pop_back(); c.n = 0; return c; }
    }
    return fresh();
  }
  // A new chunk. Pinned chunks come in slabs of pageable memory that is written once and then registered with
  // CUDA: 7 GB/s on the test nodes against 2 GB/s of cudaHostAlloc, with the same copy speed (src/proto/
  // pin_test.cpp). The other chunks of a slab go to the free list. The main thread also calls this while the
  // GPUs work, so that chunks are there before they are needed.
  static const int kSlab = 16;
  Chunk fresh() {
    const size_t bytes = (size_t)cap * (g_L.words() * sizeof(uint64_t) + sizeof(double));
    const int count = pinned ? kSlab : 1;
    void* p = nullptr;
    const auto ta = std::chrono::steady_clock::now();
    if (posix_memalign(&p, pinned ? (size_t)1 << 21 : 4096, bytes * count)) die("host allocation failed", __LINE__);
    if (pinned) {
      memset(p, 0, bytes * count);
      CUDA(cudaHostRegister(p, bytes * count, cudaHostRegisterPortable));
    }
    allocNs += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - ta).count();
    auto chunkAt = [&](int j) {
      Chunk c;
      char* q = (char*)p + bytes * j;
      c.xz = (uint64_t*)q; c.cf = (double*)(q + (size_t)cap * g_L.words() * sizeof(uint64_t));
      c.writers = new std::atomic<int>(0);
      return c;
    };
    std::lock_guard<std::mutex> lk(m);
    allocated += count;
    for (int j = 1; j < count; j++) freeList.push_back(chunkAt(j));
    return chunkAt(0);
  }
  void put(Chunk c) { std::lock_guard<std::mutex> lk(m); c.n = 0; freeList.push_back(c); }
};
static ChunkPool g_pool;

struct Bucket { std::vector<Chunk> ch; long n = 0; };
struct Work { Bucket b; int startOp; int pendingTrunc = 0; bool onGpu = false; };   // pendingTrunc: operation startOp-1 was applied, its deduplication and truncation were not; onGpu: the terms are the worker's current expansion (b.n of them), not host chunks
// pendingTrunc 2 (--fixed-rows): the bucket may hold equal strings from the exchange after a changed row; merge before operation startOp

// Reserves room for c terms at the tail of a bucket (caller holds the bucket's lock if shared).
static void reserveTail(Bucket& b, long c, std::vector<Seg>& out) {
  const int W = g_L.words();
  while (c > 0) {
    if (b.ch.empty() || b.ch.back().n == g_pool.cap) b.ch.push_back(g_pool.get());
    Chunk& k = b.ch.back();
    const long take = std::min(c, g_pool.cap - k.n);
    k.writers->fetch_add(1);
    out.push_back({k.xz + (size_t)k.n * W, k.cf + k.n, take, k.writers});
    k.n += take; b.n += take; c -= take;
  }
}
static void appendHost(Bucket& b, const uint64_t* xz, const double* cf, long c) {
  std::vector<Seg> segs;
  reserveTail(b, c, segs);
  const int W = g_L.words();
  long off = 0;
  for (auto& s : segs) {
    memcpy(s.xz, xz + (size_t)off * W, (size_t)s.n * W * sizeof(uint64_t));
    memcpy(s.cf, cf + off, (size_t)s.n * sizeof(double));
    off += s.n;
    s.writers->fetch_sub(1);
  }
}
static void releaseBucket(Bucket& b) { for (auto& c : b.ch) g_pool.put(c); b.ch.clear(); b.n = 0; }

// ---------------------------------------------------------------- windows

struct Window {
  int b = 0, e = 0;                 // operations [b, e) in processing order
  int g = 0;                        // 2^g buckets
  std::vector<uint64_t> rows;       // g masks, valid for strings at position b
  std::vector<BV> spare;            // further rows of the same window, for splitting
  int annDim = 0, generators = 0;
  double cosetFrac = 0;             // share of the sample in the largest coset of the generator span
  int lastBreaks = 0;               // --fixed-rows: the last operation changes a fixed row; its truncation waits for the exchange
};

// --fixed-rows 1 (rank 0): the rows chosen for the first window that divides the terms, kept for the rest of the run. A later
// window ends with the first branching rotation that changes one of the rows in use; that rotation is applied without its
// truncation, the terms go to their buckets by the same rows at the boundary, and the next window merges and truncates first.
static std::vector<BV> g_fixed;
static Window planFixed(int b, int gWanted, const Opts& o, std::mt19937_64& rng);

static int ownerCpu(const uint64_t* t, const std::vector<uint64_t>& rows, int g) {
  int ow = 0;
  for (int j = 0; j < g; j++) ow |= dotParity(t, &rows[(size_t)j * g_L.words()], g_L.words()) << j;
  return ow;
}

// Share of the sample that falls into the largest coset of V. Buckets are unions of cosets, so this
// bounds how evenly any choice of rows can split the terms.
static double largestCosetShare(const std::vector<uint64_t>& sample, const Span& V, size_t maxTerms) {
  const int W = g_L.words();
  const size_t ns = sample.size() / W;
  if (!ns) return 0;
  const size_t stride = std::max<size_t>(1, ns / maxTerms);
  std::vector<uint64_t> key;
  BV t(W);
  for (size_t i = 0; i < ns; i += stride) {
    std::copy(sample.begin() + i * W, sample.begin() + (i + 1) * W, t.begin());
    V.reduce(t);
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int w = 0; w < W; w++) { h ^= t[w] + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 31; }
    key.push_back(h);
  }
  std::sort(key.begin(), key.end());
  size_t best = 0, run = 0;
  for (size_t i = 0; i < key.size(); i++) { run = (i && key[i] == key[i - 1]) ? run + 1 : 1; best = std::max(best, run); }
  return (double)best / (double)key.size();
}

// Finds the window that starts at b and the rows that survive it.
//   sample: strings as they will be at position b (may be empty); gWanted: desired log2(buckets).
static Window planWindow(int b, int gWanted, const std::vector<uint64_t>& sample, const Opts& o, std::mt19937_64& rng) {
  if (o.fixedRows && !g_fixed.empty()) return planFixed(b, gWanted, o, rng);
  const int W = g_L.words(), M = (int)g_ops.size(), n2 = 2 * g_L.n;
  gWanted = std::min(gWanted, o.gmax);
  const bool cosetRule = (o.window == "coset") && gWanted > 0 && !sample.empty();
  const int annMin = std::min(n2 - 1, std::max(o.annMin < 0 ? g_L.n : o.annMin, 1));
  const int dimCap = cosetRule ? n2 - (gWanted + 4) : n2 - annMin;

  // Generators of the branching rotations ahead, as seen from position b, up to the dimension cap.
  struct Gen { int k; BV v; bool fresh; };
  std::vector<Gen> gens;
  std::vector<int> maps;
  Span V; V.L = g_L;
  int k = b;
  for (; k < M; k++) {
    const Op& op = g_ops[k];
    if (op.type == OP_ROT && op.branches) {
      BV v = op.gen;
      for (int j = (int)maps.size() - 1; j >= 0; --j) mapVec(g_ops[maps[j]], v.data(), g_L);
      BV t = v;
      const bool fresh = V.reduce(t);
      if (fresh) { if (V.dim() + 1 > dimCap) break; V.add(v); }
      gens.push_back({k, v, fresh});
    } else if (changesBits(op)) {
      maps.push_back(k);
    }
  }
  int e = k;

  Window w; w.b = b;
  if (cosetRule) {
    // Longest prefix whose generator span still leaves the sample spread over many cosets. The share
    // of the largest coset only grows as generators are added, so the last passing prefix is kept.
    const double limit = 1.0 / (o.kappa * (double)(1 << gWanted));
    Span P; P.L = g_L;
    int passEnd = -1;                       // index into gens after which the window may end
    size_t i = 0;
    for (; i < gens.size(); i++) {
      if (gens[i].fresh) {
        P.add(gens[i].v);
        if (P.dim() % 8 == 0 || P.dim() == V.dim()) {
          if (largestCosetShare(sample, P, 20000) > limit) break;
          passEnd = (int)i;
        }
      }
    }
    if (i < gens.size()) {
      // Cut back to the last passing point (at least the first generator), ending before the next one.
      const int keep = std::max(passEnd, 0);
      e = (keep + 1 < (int)gens.size()) ? gens[keep + 1].k : e;
      gens.resize(keep + 1);
    }
    V = Span(); V.L = g_L;
    for (auto& gn : gens) V.add(gn.v);
    w.cosetFrac = largestCosetShare(sample, V, 20000);
    while (gWanted > 0 && w.cosetFrac > 1.0 / (double)(1 << gWanted)) gWanted--;   // cannot split finer than the cosets
  }
  w.e = e;
  w.generators = (int)gens.size();
  std::vector<BV> ann = V.annihilator();
  w.annDim = (int)ann.size();
  w.g = std::max(0, std::min(gWanted, w.annDim));

  std::mt19937_64 matchedRng;
  std::mt19937_64& rowRng = o.policySeed < 0 ? rng : matchedRng;
  auto randomRow = [&]() {
    BV r(W, 0);
    bool any = false;
    while (!any)
      for (auto& a : ann) if (rowRng() & 1ULL) { any = true; for (int x = 0; x < W; x++) r[x] ^= a[x]; }
    return r;
  };
  const size_t ns = sample.size() / W;
  double bestMax = 1e300;
  for (int c = 0; c < (w.g && ns ? o.cands : 1); c++) {
    if (o.policySeed >= 0) matchedRng.seed(policyStreamSeed(o.policySeed, b, (uint64_t)c));
    std::vector<uint64_t> rows((size_t)std::max(w.g, 1) * W, 0);
    for (int j = 0; j < w.g; j++) { BV r = randomRow(); std::copy(r.begin(), r.end(), rows.begin() + (size_t)j * W); }
    double mx = 0;
    if (w.g && ns) {
      std::vector<long> h((size_t)1 << w.g, 0);
      for (size_t i = 0; i < ns; i++) h[ownerCpu(&sample[i * W], rows, w.g)]++;
      mx = (double)*std::max_element(h.begin(), h.end());
    }
    if (mx < bestMax) { bestMax = mx; w.rows = rows; }
  }
  if (o.policySeed >= 0) matchedRng.seed(policyStreamSeed(o.policySeed, b, 0x100000000ULL));
  if (!ann.empty()) for (int i = 0; i < 32; i++) w.spare.push_back(randomRow());
  if (o.fixedRows && g_fixed.empty() && w.g > 0) {
    // The rows for the rest of the run: as many as the largest bucket count, the best-balanced set on this sample.
    const int G = std::min(o.gmax, (int)ann.size());
    std::vector<uint64_t> best;
    double bestMx = 1e300;
    for (int c = 0; c < std::max(1, o.cands); c++) {
      std::vector<uint64_t> rows((size_t)G * W, 0);
      for (int j = 0; j < G; j++) { BV r = randomRow(); std::copy(r.begin(), r.end(), rows.begin() + (size_t)j * W); }
      std::vector<long> h((size_t)1 << G, 0);
      for (size_t i = 0; i < ns; i++) h[ownerCpu(&sample[i * W], rows, G)]++;
      const double mx = ns ? (double)*std::max_element(h.begin(), h.end()) : 0;
      if (mx < bestMx) { bestMx = mx; best = rows; }
    }
    for (int j = 0; j < G; j++) g_fixed.emplace_back(best.begin() + (size_t)j * W, best.begin() + (size_t)(j + 1) * W);
    for (int j = 0; j < w.g; j++) std::copy(g_fixed[j].begin(), g_fixed[j].end(), w.rows.begin() + (size_t)j * W);
    if (g_rank == 0) printf("fixed rows: %d chosen at operation %d (largest of %d buckets on the sample: %.3f of the mean)\n",
                            G, b, 1 << G, ns ? bestMx * (double)(1 << G) / (double)ns : 0.0);
  }
  return w;
}

static Window planFixed(int b, int gWanted, const Opts& o, std::mt19937_64& rng) {
  const int W = g_L.words(), M = (int)g_ops.size();
  static int lastBranch = -1;
  if (lastBranch < 0) for (int k = 0; k < M; k++) if (g_ops[k].type == OP_ROT && g_ops[k].branches) lastBranch = k;
  int g = std::max(0, std::min(std::min(gWanted, o.gmax), (int)g_fixed.size()));
  Window w; w.b = b;
  Span V; V.L = g_L;
  std::vector<int> maps;
  int k = b, gens = 0;
  for (; k < M; k++) {
    const Op& op = g_ops[k];
    if (op.type == OP_ROT && op.branches) {
      BV v = op.gen;
      for (int j = (int)maps.size() - 1; j >= 0; --j) mapVec(g_ops[maps[j]], v.data(), g_L);
      bool hit = false;
      for (int j = 0; j < g && !hit; j++) hit = dotParity(g_fixed[j].data(), v.data(), W) != 0;
      if (hit && k + 1 == M) {
        // The last operation of the circuit: no exchange follows it. It runs in a window of its own without rows (one
        // bucket), so that its truncation sees the merged terms.
        if (k > b) break;                               // this window ends before it
        g = 0;
        V.add(v);
        gens++;
        continue;
      }
      if (hit) { w.lastBreaks = 1; k++; break; }       // the last operation of this window
      V.add(v);
      gens++;
    } else if (changesBits(op)) {
      if (k < lastBranch) die("--fixed-rows: an operation that changes the bits comes before the last branching rotation", __LINE__);
      maps.push_back(k);
    }
  }
  w.e = k;
  w.generators = gens;
  std::vector<BV> ann = V.annihilator();
  w.annDim = (int)ann.size();
  w.g = g;
  w.rows.assign((size_t)std::max(g, 1) * W, 0);
  for (int j = 0; j < g; j++) std::copy(g_fixed[j].begin(), g_fixed[j].end(), w.rows.begin() + (size_t)j * W);
  // Rows for splitting a bucket inside the window: valid for every operation before the last one.
  for (int i = 0; i < 32 && !ann.empty(); i++) {
    BV r(W, 0);
    bool any = false;
    while (!any)
      for (auto& a : ann) if (rng() & 1ULL) { any = true; for (int x = 0; x < W; x++) r[x] ^= a[x]; }
    w.spare.push_back(r);
  }
  return w;
}

// Without a pilot: end the window right after the last truncation in its second half.
static int snapEnd(const Window& w) {
  const int M = (int)g_ops.size();
  if (w.e >= M) return w.e;
  for (int k = w.e - 1; k >= w.b + (w.e - w.b) / 2; --k) if (g_ops[k].trunc) return k + 1;
  return w.e;
}

static void packWindow(const Window& w, int hdr[6], std::vector<uint64_t>& payload) {
  const int W = g_L.words();
  hdr[0] = w.b; hdr[1] = w.e; hdr[2] = w.g; hdr[3] = w.annDim; hdr[4] = w.generators; hdr[5] = (int)w.spare.size();
  payload = w.rows;
  payload.resize((size_t)std::max(w.g, 1) * W, 0);
  for (auto& r : w.spare) payload.insert(payload.end(), r.begin(), r.end());
}
static Window unpackWindow(const int hdr[6], const std::vector<uint64_t>& payload) {
  const int W = g_L.words();
  Window w;
  w.b = hdr[0]; w.e = hdr[1]; w.g = hdr[2]; w.annDim = hdr[3]; w.generators = hdr[4];
  const size_t nr = (size_t)std::max(w.g, 1) * W;
  w.rows.assign(payload.begin(), payload.begin() + nr);
  for (int i = 0; i < hdr[5]; i++) w.spare.emplace_back(payload.begin() + nr + (size_t)i * W, payload.begin() + nr + (size_t)(i + 1) * W);
  return w;
}

// Rank 0 plans the window from the samples of all nodes and broadcasts it.
static Window planShared(int b, int gWanted, std::vector<uint64_t>& localSample, const Opts& o, std::mt19937_64& rng) {
  const int W = g_L.words();
  std::vector<uint64_t> sample;
  if (g_R > 1) {
    int mine = (int)localSample.size();
    std::vector<int> cnt(g_R, 0), dsp(g_R, 0);
    MPI_Gather(&mine, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (g_rank == 0) { int tot = 0; for (int r = 0; r < g_R; r++) { dsp[r] = tot; tot += cnt[r]; } sample.resize(tot); }
    MPI_Gatherv(localSample.data(), mine, MPI_UINT64_T, sample.data(), cnt.data(), dsp.data(), MPI_UINT64_T, 0, MPI_COMM_WORLD);
  } else {
    sample.swap(localSample);
  }
  Window w;
  int hdr[6] = {0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> payload;
  double frac = 0;
  if (g_rank == 0) { w = planWindow(b, gWanted, sample, o, rng); packWindow(w, hdr, payload); frac = w.cosetFrac; }
  if (g_R > 1) {
    MPI_Bcast(hdr, 6, MPI_INT, 0, MPI_COMM_WORLD);
    const int len = (std::max(hdr[2], 1) + hdr[5]) * W;
    payload.resize(len);
    MPI_Bcast(payload.data(), len, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&frac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (g_rank != 0) { w = unpackWindow(hdr, payload); w.cosetFrac = frac; }
    if (o.fixedRows) { int lb = w.lastBreaks; MPI_Bcast(&lb, 1, MPI_INT, 0, MPI_COMM_WORLD); w.lastBreaks = lb; }
  }
  return w;
}

// After a window: bucket d lives on node d mod g_R. Every node sends its pieces of other nodes'
// buckets chunk by chunk (all chunks full except the last), and the owner adopts each message as a
// chunk of its bucket. The chunks travel in rounds: round q carries the q-th chunk of every
// (sender, bucket) pair. A node posts the receives and the sends of one round, waits for them, and
// returns the chunks it has sent to the pool before the next round, so that the chunks arriving in the
// next round reuse them. The requests in flight are then bounded by the number of buckets (not by the
// number of chunks), and a node holds its own output plus the arrivals of one round (not its output
// plus all arrivals).
static double g_xCounts = 0, g_xPost = 0, g_xAlloc = 0, g_xWait = 0;   // parts of the exchange time on this node
static long g_xNewChunks = 0;                                          // chunks the last exchange had to allocate for arriving terms
static long g_xRounds = 0;                                             // rounds of the last exchange
static void exchangeBuckets(std::vector<Bucket>& nb, double& seconds, double& bytes) {
  if (g_R == 1) return;
  const double t0 = MPI_Wtime();
  const int B = (int)nb.size(), W = g_L.words();
  const int64_t cap = (int64_t)g_pool.cap;
  std::vector<int64_t> mine(B), all((size_t)B * g_R);
  for (int d = 0; d < B; d++) mine[d] = (int64_t)nb[d].n;
  MPI_Allgather(mine.data(), B, MPI_INT64_T, all.data(), B, MPI_INT64_T, MPI_COMM_WORLD);
  const double t1 = MPI_Wtime();
  const long long alloc0 = g_pool.allocNs.load();
  const long chunks0 = g_pool.total();
  // What this node sends: the non-empty chunks of every bucket of another node, in order.
  std::vector<std::vector<Chunk>> out(B);
  for (int d = 0; d < B; d++) {
    if (d % g_R == g_rank) continue;
    for (auto& k : nb[d].ch) { if (k.n > 0) out[d].push_back(k); else g_pool.put(k); }
    nb[d].ch.clear(); nb[d].n = 0;
    if ((int64_t)out[d].size() != (mine[d] + cap - 1) / cap) die("exchange: a bucket piece is not in full chunks", __LINE__);
  }
  int64_t rounds = 0;                       // the same on every node
  for (int r = 0; r < g_R; r++)
    for (int d = 0; d < B; d++)
      if (d % g_R != r) rounds = std::max(rounds, (all[(size_t)r * B + d] + cap - 1) / cap);
  double tPost = 0, tWait = 0;
  std::vector<MPI_Request> req;
  for (int64_t q = 0; q < rounds; q++) {
    const double ta = MPI_Wtime();
    req.clear();
    for (int d = 0; d < B; d++) {
      if (d % g_R != g_rank) continue;
      for (int r = 0; r < g_R; r++) {
        if (r == g_rank) continue;
        const int64_t c = all[(size_t)r * B + d] - q * cap;
        if (c <= 0) continue;
        const long take = (long)std::min<int64_t>(c, cap);
        Chunk k = g_pool.get();
        k.n = take;
        req.emplace_back(); MPI_Irecv(k.xz, (int)(take * W), MPI_UINT64_T, r, 2 * d, MPI_COMM_WORLD, &req.back());
        req.emplace_back(); MPI_Irecv(k.cf, (int)take, MPI_DOUBLE, r, 2 * d + 1, MPI_COMM_WORLD, &req.back());
        nb[d].ch.push_back(k); nb[d].n += take;
      }
    }
    for (int d = 0; d < B; d++) {
      if (d % g_R == g_rank || q >= (int64_t)out[d].size()) continue;
      Chunk& k = out[d][(size_t)q];
      req.emplace_back(); MPI_Isend(k.xz, (int)(k.n * W), MPI_UINT64_T, d % g_R, 2 * d, MPI_COMM_WORLD, &req.back());
      req.emplace_back(); MPI_Isend(k.cf, (int)k.n, MPI_DOUBLE, d % g_R, 2 * d + 1, MPI_COMM_WORLD, &req.back());
      bytes += (double)k.n * (W * 8 + 8);
    }
    const double tb = MPI_Wtime();
    if (!req.empty()) MPI_Waitall((int)req.size(), req.data(), MPI_STATUSES_IGNORE);
    for (int d = 0; d < B; d++)
      if (d % g_R != g_rank && q < (int64_t)out[d].size()) g_pool.put(out[d][(size_t)q]);
    const double tc = MPI_Wtime();
    tPost += tb - ta; tWait += tc - tb;
  }
  g_xNewChunks = g_pool.total() - chunks0;
  g_xRounds = (long)rounds;
  const double t3 = MPI_Wtime();
  g_xCounts += t1 - t0; g_xPost += tPost; g_xWait += tWait;
  g_xAlloc += (double)(g_pool.allocNs.load() - alloc0) * 1e-9;
  seconds += t3 - t0;
}

// Optional (--stream 1): while the workers of this node run the window, the main thread
// ships every finished chunk of other nodes' buckets and adopts the chunks that arrive for its own
// buckets, so the exchange overlaps the computation instead of following it. `finished` counts the
// workers that have returned. Each chunk travels as two messages with the bucket id as tag (bits,
// then coefficients); a final message per peer says how many chunks were sent to it.
static void streamExchange(std::vector<Bucket>& nb, std::vector<std::mutex>& bm, const std::atomic<int>& finished, int nWorkers,
                           double& tailSeconds, double& bytes) {
  const int B = (int)nb.size(), W = g_L.words(), END = 1000;
  struct Flight { MPI_Request r[2]; Chunk c; };
  std::vector<Flight> flights;
  std::vector<long> sentTo(g_R, 0), recvFrom(g_R, 0), expectFrom(g_R, -1);
  bool endSent = false;
  double tDone = -1;
  auto shipReady = [&](bool flushAll) {
    for (int d = 0; d < B; d++) {
      if (d % g_R == g_rank) continue;
      std::vector<Chunk> ready;
      {
        std::lock_guard<std::mutex> lk(bm[d]);
        auto& ch = nb[d].ch;
        for (size_t i = 0; i < ch.size();) {
          const bool sealed = ch[i].writers->load() == 0 && (flushAll || ch[i].n == g_pool.cap);
          if (sealed) { ready.push_back(ch[i]); nb[d].n -= ch[i].n; ch.erase(ch.begin() + i); } else i++;
        }
      }
      for (auto& k : ready) {
        if (k.n == 0) { g_pool.put(k); continue; }
        flights.emplace_back();
        Flight& f = flights.back(); f.c = k;
        MPI_Isend(k.xz, (int)(k.n * W), MPI_UINT64_T, d % g_R, d, MPI_COMM_WORLD, &f.r[0]);
        MPI_Isend(k.cf, (int)k.n, MPI_DOUBLE, d % g_R, d, MPI_COMM_WORLD, &f.r[1]);
        sentTo[d % g_R]++;
        bytes += (double)k.n * (W * 8 + 8);
      }
    }
  };
  auto retire = [&]() {
    for (size_t i = 0; i < flights.size();) {
      int done = 0;
      MPI_Testall(2, flights[i].r, &done, MPI_STATUSES_IGNORE);
      if (done) { g_pool.put(flights[i].c); flights[i] = flights.back(); flights.pop_back(); } else i++;
    }
  };
  auto receive = [&]() {
    while (true) {
      int flag = 0; MPI_Status st;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &st);
      if (!flag) return;
      if (st.MPI_TAG == END) {
        long n = 0;
        MPI_Recv(&n, 1, MPI_LONG, st.MPI_SOURCE, END, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        expectFrom[st.MPI_SOURCE] = n;
        continue;
      }
      int cnt = 0;
      MPI_Get_count(&st, MPI_UINT64_T, &cnt);
      const int d = st.MPI_TAG;
      Chunk k = g_pool.get();
      k.n = cnt / W;
      MPI_Recv(k.xz, cnt, MPI_UINT64_T, st.MPI_SOURCE, d, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(k.cf, (int)k.n, MPI_DOUBLE, st.MPI_SOURCE, d, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      { std::lock_guard<std::mutex> lk(bm[d]); nb[d].ch.push_back(k); nb[d].n += k.n; }
      recvFrom[st.MPI_SOURCE]++;
    }
  };
  while (true) {
    const bool workersDone = finished.load() == nWorkers;
    if (workersDone && tDone < 0) tDone = MPI_Wtime();
    shipReady(workersDone);
    receive();
    retire();
    if (workersDone && !endSent) {
      shipReady(true);
      for (int r = 0; r < g_R; r++) if (r != g_rank) MPI_Send(&sentTo[r], 1, MPI_LONG, r, END, MPI_COMM_WORLD);
      endSent = true;
    }
    if (endSent && flights.empty()) {
      bool all = true;
      for (int r = 0; r < g_R; r++) if (r != g_rank && recvFrom[r] != expectFrom[r]) all = false;
      if (all) break;
    }
    if (!workersDone) std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  tailSeconds += MPI_Wtime() - tDone;
}

// ---------------------------------------------------------------- workers

struct Shared {
  std::mutex qm;
  std::condition_variable qcv;
  std::deque<Work> queue;
  int active = 0;
  std::vector<Bucket> nextB;
  std::vector<std::mutex> bm{4096};
  std::mutex em;
  double expec = 0;
  std::atomic<long> nout{0}, peakSum{0}, splits{0}, runs{0}, peak{0}, hostSplits{0}, verified{0}, violations{0}, earlyDedups{0}, shares{0};
  std::atomic<int> waiting{0};          // workers of this node with nothing to do while another one is busy
  std::atomic<int> finished{0};
  // conserved rows as masks for strings at the start of the current window, and their expected values
  std::vector<uint64_t> checkRows;
  int checkG = 0, checkExpected = 0;
  std::atomic<long>* popAt = nullptr;   // terms seen before each operation, summed over buckets
  // One GPU of one node: the operator can stay on the GPU between windows.
  std::atomic<bool> canStay{false};     // this window is one bucket and not the last window
  std::atomic<bool> stayed{false};      // ... and its result stayed on the GPU
  int nextRows = 0;                     // rows planned for the next window
  // The rows of the next window may be chosen while this window runs; a worker waits for them before it partitions.
  std::mutex planM;
  std::condition_variable planCv;
  bool planReady = true;
  long oneBucket = 0;                   // target size: up to this many terms at a window start are one bucket
  bool splitEarly = false;              // a single bucket far above the target size is split when the window starts
};

#include "noise_fusion.hpp"

struct Worker {
  int dev = 0;
  Opts o;
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  size_t xzPerTerm = 0, xzMem = 0, cfMem = 0, wsMem = 0;
  long capTerms = 0;
  void *xz[2], *cf[2], *dWs;
  uint16_t* dOwners;
  int32_t* dIdx;
  cupaulipropPauliExpansion_t ex[2];
  bool haveEx = false;
  int cur = 0;
  cupaulipropWorkspaceDescriptor_t ws;
  cupaulipropCoefficientTruncationParams_t coefParams, junkParams;
  cupaulipropPauliWeightTruncationParams_t weightParams;
  std::vector<cupaulipropTruncationStrategy_t> trunc, junk;
  std::vector<cupaulipropQuantumOperator_t> oper;
  std::mt19937_64 rng;
  double tOps = 0, tPart = 0, tH2D = 0, tD2H = 0, tIdle = 0, tVerify = 0, tPlanWait = 0;
  NoiseFusion noiseFusion;
  bool onGpu = false;              // the current expansion is the whole operator at a window boundary
  long windowsOnGpu = 0;

  void init(int device, const Opts& opts) {
    dev = device; o = opts;
    CUDA(cudaSetDevice(dev));
    CUPP(cupaulipropCreate(&handle));
    xzPerTerm = g_L.words() * sizeof(uint64_t);
    const size_t perTerm = xzPerTerm + sizeof(double);
    size_t freeMem = 0, totalMem = 0;
    CUDA(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t usable = (size_t)(freeMem * o.vramFrac);
    capTerms = (long)((size_t)(usable * o.expShare) / perTerm);
    xzMem = (size_t)capTerms * xzPerTerm; cfMem = (size_t)capTerms * sizeof(double);
    wsMem = usable - 2 * (xzMem + cfMem) - (size_t)capTerms * 12;
    for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&xz[b], xzMem)); CUDA(cudaMalloc(&cf[b], cfMem)); }
    CUDA(cudaMalloc(&dWs, wsMem));
    CUDA(cudaMalloc((void**)&dOwners, (size_t)capTerms * sizeof(uint16_t)));
    CUDA(cudaMalloc((void**)&dIdx, (size_t)capTerms * sizeof(int32_t)));
    CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));
    noiseFusion.init(handle, kNoiseGroup, false);
    coefParams.cutoff = o.coef; weightParams.cutoff = o.weight; junkParams.cutoff = o.junk;
    trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
    if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});
    junk.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &junkParams});
    rng.seed((uint64_t)o.seed * 7919ULL + dev + 131ULL * g_rank);
    oper.resize(g_ops.size());
    for (size_t i = 0; i < g_ops.size(); i++) {
      const Op& p = g_ops[i];
      if (p.type == OP_ROT)
        CUPP(cupaulipropCreatePauliRotationGateOperator(handle, p.angle, (int32_t)p.qubits.size(), p.qubits.data(), p.paulis.data(), &oper[i]));
      else if (p.type == OP_CLIFF)
        CUPP(cupaulipropCreateCliffordGateOperator(handle, p.cliff, p.qubits.data(), &oper[i]));
      else
        CUPP(cupaulipropCreatePauliNoiseChannelOperator(handle, (int32_t)p.qubits.size(), p.qubits.data(), p.prob, &oper[i]));
    }
  }

  void makeExpansions(long nCur, int hasDuplicates = 0) {
    if (haveEx) { CUPP(cupaulipropDestroyPauliExpansion(ex[0])); CUPP(cupaulipropDestroyPauliExpansion(ex[1])); }
    cur = 0;
    CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[0], xzMem, cf[0], cfMem,
                                         CUDA_R_64F, nCur, CUPAULIPROP_SORT_ORDER_NONE, hasDuplicates, &ex[0]));
    CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[1], xzMem, cf[1], cfMem,
                                         CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1]));
    haveEx = true;
  }
  long numTerms() { int64_t n = 0; CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &n)); return (long)n; }

  // A spare row of the window, as a mask for strings at position k.
  BV splitRow(const Window& w, int k) {
    if (w.spare.empty()) die("window has no row left to split a bucket", __LINE__);
    BV r = w.spare[rng() % w.spare.size()];
    for (int j = w.b; j < k; j++) mapMask(g_ops[j], r.data(), g_L);
    return r;
  }

  void requeue(Shared& sh, Bucket& piece, int startOp, int pendingTrunc = 0) {
    { std::lock_guard<std::mutex> lk(sh.qm);
      sh.queue.push_front(Work());
      sh.queue.front().b.ch.swap(piece.ch); sh.queue.front().b.n = piece.n; sh.queue.front().startOp = startOp;
      sh.queue.front().pendingTrunc = pendingTrunc; }
    piece.n = 0;
    sh.qcv.notify_all();
  }

  // Copies m terms from device arrays into a bucket's reserved tail.
  void toHost(const void* srcXz, const void* srcCf, long off, const std::vector<Seg>& segs) {
    for (auto& s : segs) {
      CUDA(cudaMemcpy(s.xz, (const char*)srcXz + off * xzPerTerm, (size_t)s.n * xzPerTerm, cudaMemcpyDeviceToHost));
      CUDA(cudaMemcpy(s.cf, (const char*)srcCf + off * sizeof(double), (size_t)s.n * sizeof(double), cudaMemcpyDeviceToHost));
      off += s.n;
      s.writers->fetch_sub(1);
    }
  }

  // The operator that stayed on the GPU goes to a host bucket (the next window has several buckets and its
  // rows are scored on host terms, or a checkpoint is written).
  void toHostFromGpu(Bucket& b) {
    if (!onGpu) return;
    CUDA(cudaSetDevice(dev));
    const double t0 = now();
    std::vector<Seg> segs;
    reserveTail(b, numTerms(), segs);
    toHost(xz[cur], cf[cur], 0, segs);
    tD2H += now() - t0;
    onGpu = false;
  }

  // Runs a copy of the bucket through the window and records the number of terms after every
  // operation. Returns how many operations it completed (it stops if the bucket outgrows the GPU).
  // The pilot is the part of the bucket on which `rows` further rows of the window are all zero: a
  // smaller sub-bucket that is just as independent. `used` returns its size.
  int pilot(const Window& w, const Bucket& b, int rows, std::vector<double>& after, double& used) {
    CUDA(cudaSetDevice(dev));
    used = 0;
    if (b.n == 0 || b.n > capTerms / 2) return 0;
    long off = 0;
    for (auto& k : b.ch) {
      if (k.n == 0) continue;
      CUDA(cudaMemcpy((char*)xz[0] + off * xzPerTerm, k.xz, (size_t)k.n * xzPerTerm, cudaMemcpyHostToDevice));
      CUDA(cudaMemcpy((char*)cf[0] + off * sizeof(double), k.cf, (size_t)k.n * sizeof(double), cudaMemcpyHostToDevice));
      off += k.n;
    }
    long n0 = b.n;
    rows = std::min<int>(rows, (int)w.spare.size());
    if (rows > 0 && n0 > 100000) {
      const int words = g_L.words();
      std::vector<uint64_t> m((size_t)rows * words);
      for (int j = 0; j < rows; j++) std::copy(w.spare[j].begin(), w.spare[j].end(), m.begin() + (size_t)j * words);
      std::vector<int64_t> c((size_t)1 << rows, 0);
      pp_owners16((const uint64_t*)xz[0], n0, words, m.data(), rows, dOwners);
      if (pp_partition16((const uint64_t*)xz[0], (const double*)cf[0], dOwners, n0, words, (uint64_t*)xz[1], (double*)cf[1], dIdx) != 0) die("partition", __LINE__);
      pp_counts_sorted16(dOwners, n0, 1 << rows, c.data());
      if (c[0] > 0) {
        CUDA(cudaMemcpy(xz[0], xz[1], (size_t)c[0] * xzPerTerm, cudaMemcpyDeviceToDevice));
        CUDA(cudaMemcpy(cf[0], cf[1], (size_t)c[0] * sizeof(double), cudaMemcpyDeviceToDevice));
        n0 = c[0];
      }
    }
    used = (double)n0;
    makeExpansions(n0);
    for (int k = w.b; k < w.e; k++) {
      const Op& op = g_ops[k];
      const long m = numTerms();
      if (m == 0) return w.e - w.b;
      const bool dust = (op.type == OP_ROT && !op.branches);
      const int32_t nT = op.trunc ? (int32_t)trunc.size() : (dust ? 1 : 0);
      const cupaulipropTruncationStrategy_t* pT = op.trunc ? trunc.data() : (dust ? junk.data() : nullptr);
      cupaulipropPauliExpansionView_t v;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
      int64_t rp = 0, rc = 0, need = 0;
      cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, v, oper[k], CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, wsMem, &rp, &rc, ws);
      if (sp == CUPAULIPROP_STATUS_SUCCESS)
        CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
      if (sp != CUPAULIPROP_STATUS_SUCCESS || rp > (int64_t)xzMem || rc > (int64_t)cfMem || need > (int64_t)wsMem) {
        CUPP(cupaulipropDestroyPauliExpansionView(v));
        return k - w.b;
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, v, ex[1 - cur], oper[k], 1, CUPAULIPROP_SORT_ORDER_NONE, 0, nT, pT, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      cur ^= 1;
      after[k - w.b] = (double)numTerms();
    }
    return w.e - w.b;
  }

  // Merges equal strings of the current expansion. False if the workspace is too small for it.
  bool dedupNow() {
    const long m = numTerms();
    if (m == 0) return true;
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
    int64_t need = 0;
    cupaulipropStatus_t sd = cupaulipropPauliExpansionViewPrepareDeduplication(handle, v, CUPAULIPROP_SORT_ORDER_NONE, wsMem, ws);
    if (sd == CUPAULIPROP_STATUS_SUCCESS)
      CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
    if (sd != CUPAULIPROP_STATUS_SUCCESS || need > (int64_t)wsMem) { CUPP(cupaulipropDestroyPauliExpansionView(v)); return false; }
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
    CUPP(cupaulipropPauliExpansionViewExecuteDeduplication(handle, v, ex[1 - cur], CUPAULIPROP_SORT_ORDER_NONE, ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    cur ^= 1;
    return true;
  }
  // The scheduled truncation on an expansion whose equal strings are merged.
  void truncNow() {
    const long m = numTerms();
    if (m == 0) return;
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
    CUPP(cupaulipropPauliExpansionViewPrepareTruncation(handle, v, (int32_t)trunc.size(), trunc.data(), wsMem, ws));
    CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
    CUPP(cupaulipropPauliExpansionViewExecuteTruncation(handle, v, ex[1 - cur], (int32_t)trunc.size(), trunc.data(), ws, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    cur ^= 1;
  }
  // Splits the current expansion (strings at position k) by `bits` more rows of the window (one row: two halves)
  // and requeues the pieces.
  void splitCurrent(const Window& w, Shared& sh, int k, int pendingTrunc, bool share = false, int bits = 1) {
    sh.canStay = false;              // several pieces: they end in the next pool on the host
    const int words = g_L.words();
    const long m = numTerms();
    if (m < 1000) die("cannot split further", __LINE__);
    if (w.spare.empty()) die("window has no row left to split a bucket", __LINE__);
    bits = std::max(1, std::min<int>(bits, (int)w.spare.size()));
    std::vector<uint64_t> rows;
    const size_t first = rng() % w.spare.size();        // `bits` different spare rows, as masks for strings at position k
    for (int j = 0; j < bits; j++) {
      BV r = w.spare[(first + j) % w.spare.size()];
      for (int i = w.b; i < k; i++) mapMask(g_ops[i], r.data(), g_L);
      rows.insert(rows.end(), r.begin(), r.end());
    }
    std::vector<int64_t> c2((size_t)1 << bits, 0);
    pp_owners16((const uint64_t*)xz[cur], m, words, rows.data(), bits, dOwners);
    if (pp_partition16((const uint64_t*)xz[cur], (const double*)cf[cur], dOwners, m, words,
                     (uint64_t*)xz[1 - cur], (double*)cf[1 - cur], dIdx) != 0) die("partition", __LINE__);
    pp_counts_sorted16(dOwners, m, 1 << bits, c2.data());
    long off2 = 0;
    for (size_t h = 0; h < c2.size(); h++) {
      if (c2[h] == 0) continue;
      Bucket piece;
      std::vector<Seg> segs;
      reserveTail(piece, c2[h], segs);
      toHost(xz[1 - cur], cf[1 - cur], off2, segs);
      off2 += c2[h];
      requeue(sh, piece, k, pendingTrunc);
    }
    if (share) sh.shares++; else sh.splits++;
  }

  // A regular sample of the operator that stayed on the GPU (the rows of a window are scored on it).
  void sampleOnGpu(std::vector<uint64_t>& sample, long stride) {
    CUDA(cudaSetDevice(dev));
    const long n = numTerms(), count = (n + stride - 1) / stride;
    if (count == 0) return;
    pp_sample_terms((const uint64_t*)xz[cur], n, g_L.words(), stride, (uint64_t*)xz[1 - cur]);   // the other buffer is free between operations
    const size_t off = sample.size();
    sample.resize(off + (size_t)count * g_L.words());
    CUDA(cudaMemcpy(sample.data() + off, xz[1 - cur], (size_t)count * xzPerTerm, cudaMemcpyDeviceToHost));
  }

  void run(const Window& w, const Window* next, Shared& sh, bool allowSplit) {
    CUDA(cudaSetDevice(dev));
    while (true) {
      Work wk;
      {
        const double ti = now();
        std::unique_lock<std::mutex> lk(sh.qm);
        sh.waiting++;
        sh.qcv.wait(lk, [&] { return !sh.queue.empty() || sh.active == 0; });
        sh.waiting--;
        tIdle += now() - ti;
        if (sh.queue.empty()) { lk.unlock(); sh.qcv.notify_all(); return; }
        wk.b.ch.swap(sh.queue.front().b.ch); wk.b.n = sh.queue.front().b.n; wk.startOp = sh.queue.front().startOp;
        wk.pendingTrunc = sh.queue.front().pendingTrunc; wk.onGpu = sh.queue.front().onGpu;
        sh.queue.pop_front();
        sh.active++;
      }
      process(w, next, sh, wk, allowSplit);
      { std::lock_guard<std::mutex> lk(sh.qm); sh.active--; }
      sh.qcv.notify_all();
    }
  }

  void process(const Window& w, const Window* next, Shared& sh, Work& wk, bool allowSplit) {
    const int words = g_L.words();
    Bucket& b = wk.b;
    const long nb0 = b.n;
    if (!wk.onGpu && nb0 > capTerms / 2) {
      // Too large to load with room to work: split on the host and requeue.
      sh.canStay = false;
      if (!allowSplit) die("bucket does not fit the GPU (this mode cannot split)", __LINE__);
      BV row = splitRow(w, wk.startOp);
      Bucket h[2];
      for (auto& k : b.ch)
        for (long i = 0; i < k.n; i++)
          appendHost(h[dotParity(k.xz + (size_t)i * words, row.data(), words)], k.xz + (size_t)i * words, k.cf + i, 1);
      releaseBucket(b);
      for (int s = 0; s < 2; s++) if (h[s].n > 0) requeue(sh, h[s], wk.startOp, wk.pendingTrunc);
      sh.hostSplits++;
      return;
    }
    sh.runs++;
    double t0 = now();
    if (wk.onGpu) {
      // The result of the last window is the current expansion: nothing to load.
      if (!onGpu || !haveEx || nb0 != numTerms()) die("the operator is not on the GPU", __LINE__);
      onGpu = false;
    } else {
      long off = 0;
      for (auto& k : b.ch) {
        if (k.n == 0) continue;
        CUDA(cudaMemcpy((char*)xz[0] + off * xzPerTerm, k.xz, (size_t)k.n * xzPerTerm, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy((char*)cf[0] + off * sizeof(double), k.cf, (size_t)k.n * sizeof(double), cudaMemcpyHostToDevice));
        off += k.n;
      }
      releaseBucket(b);
      makeExpansions(nb0, wk.pendingTrunc ? 1 : 0);   // a piece with pending work may hold equal strings
      tH2D += now() - t0;
    }

    // Integrity: every term must carry the expected values of the parities the circuit conserves.
    if (sh.checkG > 0 && wk.startOp == w.b) {
      const double tv = now();
      pp_owners16((const uint64_t*)xz[cur], nb0, words, sh.checkRows.data(), sh.checkG, dOwners);
      sh.violations += pp_count_ne16(dOwners, nb0, sh.checkExpected);
      sh.verified += nb0;
      tVerify += now() - tv;
    }

    t0 = now();
    long peak = nb0;
    // One GPU: the operator stayed one bucket although it grew far above the target size (the estimate was too
    // low and the window has no rows). It is split now, where it is smallest, into buckets of the target size.
    if (sh.splitEarly && allowSplit && wk.startOp == w.b && !w.spare.empty() && nb0 > 2 * sh.oneBucket) {
      int bits = 1;
      while (bits < 8 && (nb0 >> bits) > sh.oneBucket) bits++;
      splitCurrent(w, sh, wk.startOp, wk.pendingTrunc, false, bits);
      tPart += now() - t0;
      return;
    }
    // Deferred merging: equal strings stay in this bucket for the whole window, so they need to be
    // merged only before a truncation looks at the coefficients, not after every operation.
    const int defer = o.deferDedup;
    int sinceDedup = defer > 0 ? 1 : 0;   // a loaded piece may carry unmerged strings
    bool merged = wk.onGpu;               // equal strings are known to be merged (noise channels are grouped only then)
    if (wk.pendingTrunc) {
      if (!dedupNow()) {
        if (!allowSplit) die("capacity (this mode cannot split)", __LINE__);
        splitCurrent(w, sh, wk.startOp, wk.pendingTrunc);
        tOps += now() - t0;
        return;
      }
      if (wk.pendingTrunc == 1) truncNow();
      sinceDedup = 0;
      merged = true;
    }
    for (int k = wk.startOp; k < w.e; k++) {
      const Op& op = g_ops[k];
      const long m = numTerms();
      if (m == 0) break;
      peak = std::max(peak, m);
      if (sh.popAt) sh.popAt[k] += m;
      // Truncation: the scheduled one, or only numerical dust after a Clifford-angle rotation.
      const bool dust = (op.type == OP_ROT && !op.branches);
      const bool fused = op.trunc && defer == 0 && !(w.lastBreaks && k == w.e - 1);   // --fixed-rows: after the exchange
      const int32_t nT = fused ? (int32_t)trunc.size() : (dust ? 1 : 0);
      const cupaulipropTruncationStrategy_t* pT = fused ? trunc.data() : (dust ? junk.data() : nullptr);
      cupaulipropPauliExpansionView_t v;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
      // Consecutive noise channels in one call, up to the next truncation and inside this window.
      int applied = 1;
      int limit = kNoiseGroup;
      if (defer > 0) limit = std::min(limit, defer - sinceDedup);
      const int group = (defer > 0 || merged) ? noiseFusionCount(k, w.e, limit) : 1;
      if (group > 1) applied = noiseFusion.apply(v, ex[1 - cur], oper, k, group, defer == 0, trunc, capTerms, wsMem, dWs, stream);
      if (applied < 2) {
        applied = 1;
        int64_t rp = 0, rc = 0, need = 0;
        cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
          handle, v, oper[k], CUPAULIPROP_SORT_ORDER_NONE, defer > 0 ? 1 : 0, nT, pT, wsMem, &rp, &rc, ws);
        if (sp == CUPAULIPROP_STATUS_SUCCESS)
          CUPP(cupaulipropWorkspaceGetMemorySize(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, &need));
        if (sp != CUPAULIPROP_STATUS_SUCCESS || rp > (int64_t)xzMem || rc > (int64_t)cfMem || need > (int64_t)wsMem) {
          CUPP(cupaulipropDestroyPauliExpansionView(v));
          // Unmerged strings may be what fills the buffer: merge them and try this operation again.
          if (defer > 0 && sinceDedup > 0 && dedupNow()) { sinceDedup = 0; sh.earlyDedups++; k--; continue; }
          // Outgrew this GPU: split by one more row of the window and requeue both halves.
          if (!allowSplit) { fprintf(stderr, "operation %d: %ld terms do not fit the GPU\n", k, m); die("capacity (this mode cannot split)", __LINE__); }
          splitCurrent(w, sh, k, false);
          tOps += now() - t0;
          return;
        }
        CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
        CUPP(cupaulipropPauliExpansionViewComputeOperatorApplication(
          handle, v, ex[1 - cur], oper[k], 1, CUPAULIPROP_SORT_ORDER_NONE, defer > 0 ? 1 : 0, nT, pT, ws, stream));
        CUDA(cudaStreamSynchronize(stream));
        merged = (defer == 0);
      }
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      cur ^= 1;
      if (sh.popAt) for (int j = 1; j < applied; j++) sh.popAt[k + j] += m;
      k += applied - 1;                       // the group ends at operation k
      const Op& lastOp = g_ops[k];
      if (defer > 0) {
        sinceDedup += applied;
        if (lastOp.trunc || k + 1 == w.e || sinceDedup >= defer) {
          if (!dedupNow()) {
            if (!allowSplit) die("capacity (this mode cannot split)", __LINE__);
            splitCurrent(w, sh, k + 1, lastOp.trunc);
            tOps += now() - t0;
            return;
          }
          sinceDedup = 0;
          merged = true;
          if (lastOp.trunc) truncNow();
        }
      }
      // Other GPUs of this node have nothing to do and this bucket is large: give them half of it.
      if (allowSplit && o.shareAbove > 0 && lastOp.trunc && k + 1 < w.e && sh.waiting.load() > 0 && numTerms() > o.shareAbove) {
        splitCurrent(w, sh, k + 1, false, true);
        tOps += now() - t0;
        return;
      }
    }
    long pk = sh.peak.load();
    while (peak > pk && !sh.peak.compare_exchange_weak(pk, peak)) {}
    sh.peakSum += peak;
    const long m = numTerms();
    sh.nout += m;
    tOps += now() - t0;
    if (m == 0) return;

    if (!next) {                                      // last window: trace with |0...0>
      cupaulipropPauliExpansionView_t v;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
      CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      double sig = 0, e2 = 0;
      CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &e2, ws, stream));
      CUDA(cudaStreamSynchronize(stream));
      CUPP(cupaulipropDestroyPauliExpansionView(v));
      std::lock_guard<std::mutex> lk(sh.em);
      sh.expec += sig * std::pow(2.0, e2);
      return;
    }

    // One GPU and the window ran as one bucket. The result stays on the GPU if the next window is one bucket: it
    // was planned so, or it was planned with rows from an estimate and the result is within the target size.
    { const double tw = now(); std::unique_lock<std::mutex> lk(sh.planM); sh.planCv.wait(lk, [&] { return sh.planReady; }); tPlanWait += now() - tw; }
    if (sh.canStay && (sh.nextRows == 0 || m <= sh.oneBucket)) { onGpu = true; windowsOnGpu++; sh.stayed = true; return; }

    // Partition the output by the next window's rows; copy each piece straight to its bucket.
    t0 = now();
    const int gN = next->g;
    std::vector<int64_t> cnt((size_t)1 << gN, 0);
    const void *srcXz = xz[cur], *srcCf = cf[cur];
    if (gN > 0) {
      pp_owners16((const uint64_t*)xz[cur], m, words, next->rows.data(), gN, dOwners);
      if (pp_partition16((const uint64_t*)xz[cur], (const double*)cf[cur], dOwners, m, words,
                       (uint64_t*)xz[1 - cur], (double*)cf[1 - cur], dIdx) != 0) die("partition", __LINE__);
      pp_counts_sorted16(dOwners, m, 1 << gN, cnt.data());
      srcXz = xz[1 - cur]; srcCf = cf[1 - cur];
    } else {
      cnt[0] = m;
    }
    CUDA(cudaDeviceSynchronize());
    tPart += now() - t0;
    t0 = now();
    long o2 = 0;
    for (size_t d = 0; d < cnt.size(); d++) {
      if (cnt[d] == 0) continue;
      std::vector<Seg> segs;
      { std::lock_guard<std::mutex> lk(sh.bm[d]); reserveTail(sh.nextB[d], cnt[d], segs); }
      toHost(srcXz, srcCf, o2, segs);       // the reserved ranges are private to this worker
      o2 += cnt[d];
    }
    tD2H += now() - t0;
  }
};

// ---------------------------------------------------------------- circuits

static Problem buildProblem(const Opts& o) {
  const std::string& c = o.circuit;
  if (c == "ki")       return makeKickedIsingFrame(o.steps, o.theta, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  if (c == "ki_lab")   return makeIsing("kicked Ising, Eagle heavy-hex", NUM_CIRCUIT_QUBITS, heavyHexEagle(), o.steps, o.theta, -0.5, 0.0, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  if (c == "tfim_hh")  return makeIsing("TFIM Trotter, Eagle heavy-hex", NUM_CIRCUIT_QUBITS, heavyHexEagle(), o.steps, o.theta, o.phi, 0.0, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  if (c == "tfim_sq")  return makeIsing("TFIM Trotter, square lattice", o.lx * o.ly, squareLattice(o.lx, o.ly), o.steps, o.theta, o.phi, 0.0, o.obsQ < 0 ? (o.ly / 2) * o.lx + o.lx / 2 : o.obsQ, o.truncEvery);
  if (c == "xxz")      return makeXXZ("XXZ Trotter, square lattice", o.lx * o.ly, squareLattice(o.lx, o.ly), o.steps, o.jxy, o.jz, o.obsQ < 0 ? (o.ly / 2) * o.lx + o.lx / 2 : o.obsQ, o.truncEvery);
  if (c == "qaoa")     return makeQaoa(o.n > 0 ? o.n : 100, o.layers, o.seed, o.truncEvery);
  if (c == "hea")      return makeHea(o.n > 0 ? o.n : 100, o.layers, o.seed, o.amp, o.obsQ < 0 ? (o.n > 0 ? o.n : 100) / 2 : o.obsQ, o.truncEvery);
  if (c == "ki_noisy") return makeIsing("noisy kicked Ising, Eagle heavy-hex", NUM_CIRCUIT_QUBITS, heavyHexEagle(), o.steps, o.theta, -0.5, o.noise > 0 ? o.noise : 0.005, o.obsQ < 0 ? 62 : o.obsQ, o.truncEvery);
  fprintf(stderr, "unknown circuit %s (ki, ki_lab, tfim_hh, tfim_sq, xxz, qaoa, hea, ki_noisy)\n", c.c_str());
  std::exit(1);
}

// ---------------------------------------------------------------- cuPauliProp's distributed mode
// The same operation list through the library's own multi-process execution: one rank per GPU,
// global deduplication inside every operator application. Needs CUPAULIPROP_COMM_LIB.
static int runCuppDistributed(const Opts& o, const Problem& P) {
  cudaStream_t stream = 0;
  cupaulipropHandle_t handle;
  CUPP(cupaulipropCreate(&handle));
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  if (!getenv("CUPAULIPROP_COMM_LIB")) die("CUPAULIPROP_COMM_LIB is not set", __LINE__);
  CUPP(cupaulipropResetDistributedConfiguration(handle, CUPAULIPROP_DISTRIBUTED_PROVIDER_MPI, &comm, sizeof(comm)));
  const size_t xzPerTerm = g_L.words() * sizeof(uint64_t), perTerm = xzPerTerm + sizeof(double);
  size_t freeMem = 0, totalMem = 0;
  CUDA(cudaMemGetInfo(&freeMem, &totalMem));
  const size_t usable = (size_t)(freeMem * o.vramFrac);
  const size_t capTerms = (size_t)(usable * o.expShare) / perTerm;
  const size_t xzMem = capTerms * xzPerTerm, cfMem = capTerms * sizeof(double), wsMem = usable - 2 * (xzMem + cfMem);
  void *xz[2], *cf[2], *dWs;
  for (int b = 0; b < 2; b++) { CUDA(cudaMalloc(&xz[b], xzMem)); CUDA(cudaMalloc(&cf[b], cfMem)); }
  CUDA(cudaMalloc(&dWs, wsMem));
  long n0 = 0;
  if (g_rank == 0) {
    std::vector<uint64_t> hx; std::vector<double> hc;
    for (auto& t : P.observable) { hx.insert(hx.end(), t.first.begin(), t.first.end()); hc.push_back(t.second); }
    n0 = (long)hc.size();
    CUDA(cudaMemcpy(xz[0], hx.data(), hx.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(cf[0], hc.data(), hc.size() * sizeof(double), cudaMemcpyHostToDevice));
  }
  cupaulipropPauliExpansion_t ex[2];
  CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[0], xzMem, cf[0], cfMem, CUDA_R_64F, n0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[0]));
  CUPP(cupaulipropCreatePauliExpansion(handle, g_L.n, xz[1], xzMem, cf[1], cfMem, CUDA_R_64F, 0, CUPAULIPROP_SORT_ORDER_NONE, 0, &ex[1]));
  cupaulipropWorkspaceDescriptor_t ws;
  CUPP(cupaulipropCreateWorkspaceDescriptor(handle, &ws));
  cupaulipropCoefficientTruncationParams_t coefParams, junkParams; coefParams.cutoff = o.coef; junkParams.cutoff = o.junk;
  cupaulipropPauliWeightTruncationParams_t weightParams; weightParams.cutoff = o.weight;
  std::vector<cupaulipropTruncationStrategy_t> trunc, junk;
  trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &coefParams});
  if (o.weight > 0) trunc.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_PAULI_WEIGHT_BASED, &weightParams});
  junk.push_back({CUPAULIPROP_TRUNCATION_STRATEGY_COEFFICIENT_BASED, &junkParams});
  std::vector<cupaulipropQuantumOperator_t> oper(g_ops.size());
  for (size_t i = 0; i < g_ops.size(); i++) {
    const Op& p = g_ops[i];
    if (p.type == OP_ROT) CUPP(cupaulipropCreatePauliRotationGateOperator(handle, p.angle, (int32_t)p.qubits.size(), p.qubits.data(), p.paulis.data(), &oper[i]));
    else if (p.type == OP_CLIFF) CUPP(cupaulipropCreateCliffordGateOperator(handle, p.cliff, p.qubits.data(), &oper[i]));
    else CUPP(cupaulipropCreatePauliNoiseChannelOperator(handle, (int32_t)p.qubits.size(), p.qubits.data(), p.prob, &oper[i]));
  }
  if (g_rank == 0) { printf("mode=cupp ranks=%d (one GPU each) coef=%g weight=%d %s\n", g_R, o.coef, o.weight,
                            o.cuppTuned ? "tuned: rank-local application, global deduplication before each truncation" : "default: global deduplication in every operation");
                     if (o.cuppTuned && o.cuppDedupEvery > 0) printf("tuned: additional global deduplication every %d operations\n", o.cuppDedupEvery);
                     fflush(stdout); }
  const int keepDup = o.cuppTuned ? 1 : 0;
  NoiseFusion noiseFusion;                    // the same grouping of noise channels as in the engine
  noiseFusion.init(handle, kNoiseGroup, true);
  bool merged = false;
  int sinceDedup = 0;
  // Where the time of this rank goes: the library calls by phase, and inside them the MPI calls of the library
  // (only with the profiling plugin).
  const CuppSnapshotFn snapshot = cuppProfiler();
  std::vector<double> snap(3 * CUPP_KINDS + 2, 0.0);
  auto mpiSeconds = [&]() { if (!snapshot) return 0.0; snapshot(snap.data()); double t = 0; for (int k = 0; k < CUPP_KINDS; k++) t += snap[CUPP_KINDS + k]; return t; };
  enum { PH_PREPARE = 0, PH_APPLY, PH_DEDUP, PH_TRUNC, PH_TRACE, PHASES };
  double phase[PHASES] = {0, 0, 0, 0, 0}, phaseMpi[PHASES] = {0, 0, 0, 0, 0};
  double phT = 0, phM = 0;
  auto phaseBegin = [&]() { phM = mpiSeconds(); phT = MPI_Wtime(); };
  auto phaseEnd = [&](int ph) { phase[ph] += MPI_Wtime() - phT; phaseMpi[ph] += mpiSeconds() - phM; };
  MPI_Barrier(MPI_COMM_WORLD);
  const double mpiAtStart = mpiSeconds();
  std::vector<double> snapStart = snap;
  const double epochStart = epochNow();
  const double t0 = MPI_Wtime();
  int cur = 0;
  long maxLocal = 0;
  for (size_t k = 0; k < g_ops.size(); k++) {
    const Op& op = g_ops[k];
    int64_t m = 0;
    CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m));
    maxLocal = std::max<long>(maxLocal, m);
    const bool dust = (op.type == OP_ROT && !op.branches);
    // Tuned: the coefficient cutoff is not applied inside the application (it would act on rank-local,
    // unmerged terms). The 1e-13 cutoff that removes the vanishing branch of a Clifford-angle rotation is.
    const bool fused = op.trunc && !keepDup;
    const int32_t nT = fused ? (int32_t)trunc.size() : (dust ? 1 : 0);
    const cupaulipropTruncationStrategy_t* pT = fused ? trunc.data() : (dust ? junk.data() : nullptr);
    cupaulipropPauliExpansionView_t v;
    CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
    int applied = 1;
    int limit = kNoiseGroup;
    if (keepDup && o.cuppDedupEvery > 0) limit = std::min(limit, o.cuppDedupEvery - sinceDedup);
    const int group = (keepDup || merged) ? noiseFusionCount((int)k, (int)g_ops.size(), limit) : 1;
    phaseBegin();
    if (group > 1) applied = noiseFusion.apply(v, ex[1 - cur], oper, (int)k, group, !keepDup, trunc, (long)capTerms, wsMem, dWs, stream);
    if (applied >= 2) phaseEnd(PH_APPLY);
    if (applied < 2) {
      applied = 1;
      int64_t rp = 0, rc = 0;
      phaseBegin();
      cupaulipropStatus_t sp = cupaulipropPauliExpansionViewPrepareOperatorApplication(
        handle, v, oper[k], CUPAULIPROP_SORT_ORDER_NONE, keepDup, nT, pT, wsMem, &rp, &rc, ws);
      phaseEnd(PH_PREPARE);
      if (sp != CUPAULIPROP_STATUS_SUCCESS || rp > (int64_t)xzMem || rc > (int64_t)cfMem) {
        fprintf(stderr, "[rank %d] cuPauliProp distributed mode ran out of memory at operation %zu of %zu with %ld local terms (status %d)\n",
                g_rank, k, g_ops.size(), (long)m, (int)sp);
        die("cuPauliProp distributed capacity", __LINE__);
      }
      CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
      phaseBegin();
      cupaulipropStatus_t sc = cupaulipropPauliExpansionViewComputeOperatorApplication(
        handle, v, ex[1 - cur], oper[k], 1, CUPAULIPROP_SORT_ORDER_NONE, keepDup, nT, pT, ws, stream);
      if (sc != CUPAULIPROP_STATUS_SUCCESS) {
        fprintf(stderr, "[rank %d] cuPauliProp distributed mode failed at operation %zu with %ld local terms (status %d)\n", g_rank, k, (long)m, (int)sc);
        die("cuPauliProp distributed application", __LINE__);
      }
      CUDA(cudaStreamSynchronize(stream));
      phaseEnd(PH_APPLY);
      merged = !keepDup;
    }
    CUPP(cupaulipropDestroyPauliExpansionView(v));
    cur ^= 1;
    k += applied - 1;                         // the group ends at operation k
    const Op& lastOp = g_ops[k];
    if (keepDup) sinceDedup += applied;
    if (keepDup && (lastOp.trunc || k + 1 == g_ops.size() || (o.cuppDedupEvery > 0 && sinceDedup >= o.cuppDedupEvery))) {
      sinceDedup = 0;
      // Global deduplication, then truncation on merged coefficients.
      int64_t m2 = 0;
      CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m2));
      cupaulipropPauliExpansionView_t v2;
      CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m2, &v2));
      phaseBegin();
      cupaulipropStatus_t sd = cupaulipropPauliExpansionViewPrepareDeduplication(handle, v2, CUPAULIPROP_SORT_ORDER_NONE, wsMem, ws);
      if (sd == CUPAULIPROP_STATUS_SUCCESS) {
        CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
        sd = cupaulipropPauliExpansionViewExecuteDeduplication(handle, v2, ex[1 - cur], CUPAULIPROP_SORT_ORDER_NONE, ws, stream);
      }
      if (sd != CUPAULIPROP_STATUS_SUCCESS) {
        fprintf(stderr, "[rank %d] cuPauliProp distributed mode ran out of memory in deduplication at operation %zu with %ld local terms (status %d)\n", g_rank, k, (long)m2, (int)sd);
        die("cuPauliProp distributed deduplication", __LINE__);
      }
      CUDA(cudaStreamSynchronize(stream));
      phaseEnd(PH_DEDUP);
      merged = true;
      CUPP(cupaulipropDestroyPauliExpansionView(v2));
      cur ^= 1;
      maxLocal = std::max<long>(maxLocal, m2);
      if (lastOp.trunc) {
        int64_t m3 = 0;
        CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m3));
        cupaulipropPauliExpansionView_t v3;
        CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m3, &v3));
        phaseBegin();
        CUPP(cupaulipropPauliExpansionViewPrepareTruncation(handle, v3, (int32_t)trunc.size(), trunc.data(), wsMem, ws));
        CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
        CUPP(cupaulipropPauliExpansionViewExecuteTruncation(handle, v3, ex[1 - cur], (int32_t)trunc.size(), trunc.data(), ws, stream));
        CUDA(cudaStreamSynchronize(stream));
        phaseEnd(PH_TRUNC);
        CUPP(cupaulipropDestroyPauliExpansionView(v3));
        cur ^= 1;
      }
    }
  }
  int64_t m = 0;
  CUPP(cupaulipropPauliExpansionGetNumTerms(handle, ex[cur], &m));
  cupaulipropPauliExpansionView_t v;
  CUPP(cupaulipropPauliExpansionGetContiguousRange(handle, ex[cur], 0, m, &v));
  phaseBegin();
  CUPP(cupaulipropPauliExpansionViewPrepareTraceWithZeroState(handle, v, wsMem, ws));
  CUPP(cupaulipropWorkspaceSetMemory(handle, ws, CUPAULIPROP_MEMSPACE_DEVICE, CUPAULIPROP_WORKSPACE_SCRATCH, dWs, wsMem));
  double sig = 0, e2 = 0;
  CUPP(cupaulipropPauliExpansionViewComputeTraceWithZeroState(handle, v, &sig, &e2, ws, stream));
  CUDA(cudaStreamSynchronize(stream));
  phaseEnd(PH_TRACE);
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - t0;
  const double epochEnd = epochNow();
  const double mpiTotal = mpiSeconds() - mpiAtStart;       // also fills `snap`
  long mx = 0;
  MPI_Allreduce(&maxLocal, &mx, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  if (g_rank == 0) {
    printf("expectation=%.12g\n", sig * std::pow(2.0, e2));
    printf("largest_local_expansion=%ld terms\nwall=%.3f s\n", mx, wall);
  }
  { long local[2] = {noiseFusion.calls, noiseFusion.operations}, total[2] = {0, 0};
    MPI_Reduce(local, total, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (g_rank == 0 && total[0] > 0) printf("noise channels applied in groups: %ld operations in %ld calls (summed over ranks)\n", total[1], total[0]); }
  {
    // Averages over the ranks. A rank that waits for another one inside a collective call counts that wait as
    // MPI time of the call.
    std::vector<double> local(2 * PHASES + 3 * CUPP_KINDS + 1, 0.0), sum(local.size(), 0.0);
    for (int i = 0; i < PHASES; i++) { local[i] = phase[i]; local[PHASES + i] = phaseMpi[i]; }
    for (int i = 0; i < 3 * CUPP_KINDS; i++) local[2 * PHASES + i] = snap[i] - snapStart[i];
    local[2 * PHASES + 3 * CUPP_KINDS] = mpiTotal;
    MPI_Reduce(local.data(), sum.data(), (int)local.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (g_rank == 0) {
      printf("solve_epoch=%.3f..%.3f\n", epochStart, epochEnd);
      const char* names[PHASES] = {"prepare", "application", "deduplication", "truncation", "trace"};
      printf("cupp phases, seconds per rank (of it inside MPI calls):");
      for (int i = 0; i < PHASES; i++) printf(" %s=%.3f (%.3f)", names[i], sum[i] / g_R, sum[PHASES + i] / g_R);
      printf("; profiling plugin %s\n", snapshot ? "on" : "off");
      if (snapshot) {
        double coll = 0, p2p = 0;
        for (int k = 0; k < CUPP_KINDS; k++) (CUPP_KIND_COLLECTIVE[k] ? coll : p2p) += sum[2 * PHASES + CUPP_KINDS + k];
        printf("cupp mpi, seconds per rank: collective=%.3f point_to_point=%.3f total=%.3f\n", coll / g_R, p2p / g_R, sum[2 * PHASES + 3 * CUPP_KINDS] / g_R);
        for (int k = 0; k < CUPP_KINDS; k++)
          if (sum[2 * PHASES + k] > 0)
            printf("cupp mpi %s: calls/rank=%.0f (%.2f per operation) seconds/rank=%.3f GB/rank=%.3f\n", CUPP_KIND_NAMES[k],
                   sum[2 * PHASES + k] / g_R, sum[2 * PHASES + k] / g_R / (double)g_ops.size(),
                   sum[2 * PHASES + CUPP_KINDS + k] / g_R, sum[2 * PHASES + 2 * CUPP_KINDS + k] / g_R / 1e9);
      }
    }
  }
  MPI_Finalize();
  return 0;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
  // cuPauliProp's distributed mode hands GPU pointers to MPI: Cray MPICH must see this rank's device
  // before MPI_Init.
  bool cupp = false;
  for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--mode") && !strcmp(argv[i + 1], "cupp")) cupp = true;
  if (cupp) {
    const char* lid = getenv("SLURM_LOCALID");
    int nd0 = 0;
    if (lid && cudaGetDeviceCount(&nd0) == cudaSuccess && nd0 > 0) {
      cudaSetDevice(atoi(lid) % nd0);
      void* warm = nullptr;
      if (cudaMalloc(&warm, 1 << 20) == cudaSuccess) cudaFree(warm);
    }
  }
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_R);
  Opts o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) die("missing option value", __LINE__); return argv[++i]; };
    if      (a == "--circuit")     o.circuit = next();
    else if (a == "--mode")        o.mode = next();
    else if (a == "--gpus")        o.gpus = atoi(next());
    else if (a == "--frame")       o.frame = atoi(next());
    else if (a == "--steps")       o.steps = atoi(next());
    else if (a == "--layers")      o.layers = atoi(next());
    else if (a == "--lx")          o.lx = atoi(next());
    else if (a == "--ly")          o.ly = atoi(next());
    else if (a == "--n")           o.n = atoi(next());
    else if (a == "--obs-qubit")   o.obsQ = atoi(next());
    else if (a == "--obs-all")     o.obsAll = atoi(next());
    else if (a == "--seed")        o.seed = atoi(next());
    else if (a == "--theta")       o.theta = atof(next());
    else if (a == "--phi")         o.phi = atof(next());
    else if (a == "--jxy")         o.jxy = atof(next());
    else if (a == "--jz")          o.jz = atof(next());
    else if (a == "--noise")       o.noise = atof(next());
    else if (a == "--amp")         o.amp = atof(next());
    else if (a == "--coef")        o.coef = atof(next());
    else if (a == "--weight")      o.weight = atoi(next());
    else if (a == "--trunc-every") o.truncEvery = atoi(next());
    else if (a == "--vram-frac")   o.vramFrac = atof(next());
    else if (a == "--exp-share")   o.expShare = atof(next());
    else if (a == "--target")      o.target = atol(next());
    else if (a == "--gmax")        o.gmax = atoi(next());
    else if (a == "--cands")       o.cands = atoi(next());
    else if (a == "--fixed-sectors") o.fixedSectors = atoi(next());
    else if (a == "--policy-seed") o.policySeed = atol(next());
    else if (a == "--policy-audit") o.policyAudit = atoi(next());
    else if (a == "--ann-min")     o.annMin = atoi(next());
    else if (a == "--window")      o.window = next();
    else if (a == "--kappa")       o.kappa = atof(next());
    else if (a == "--snap")        o.snap = atoi(next());
    else if (a == "--pilot")       o.pilot = atoi(next());
    else if (a == "--fixed-rows")  o.fixedRows = atoi(next());
    else if (a == "--pilot-rows")  o.pilotRows = atoi(next());
    else if (a == "--per-gpu")     o.perGpu = atoi(next());
    else if (a == "--stream")      o.stream = atoi(next());
    else if (a == "--min-bucket")  o.minBucket = atol(next());
    else if (a == "--growth-cap")  o.growthCap = atof(next());
    else if (a == "--early-per-node") o.earlyPerNode = atoi(next());
    else if (a == "--min-bucket-early") o.minBucketEarly = atol(next());
    else if (a == "--min-bucket-early-end") o.minBucketEarlyEnd = atol(next());
    else if (a == "--share-above") o.shareAbove = atol(next());
    else if (a == "--trace")       o.trace = atoi(next());
    else if (a == "--chunk")       o.chunk = atol(next());
    else if (a == "--pinned")      o.pinned = atoi(next());
    else if (a == "--overlap-plan") o.overlapPlan = atoi(next());
    else if (a == "--long-window") o.longWindow = atof(next());
    else if (a == "--verify")      o.verify = atoi(next());
    else if (a == "--ckpt-window") o.ckptWindow = atoi(next());
    else if (a == "--ckpt-file")   o.ckptFile = next();
    else if (a == "--stop-at-ckpt") o.stopAtCkpt = atoi(next());
    else if (a == "--resume")      o.resume = next();
    else if (a == "--profile")     o.profile = next();
    else if (a == "--defer-dedup") o.deferDedup = atoi(next());
    else if (a == "--cupp-tuned")  o.cuppTuned = atoi(next());
    else if (a == "--cupp-dedup-every") o.cuppDedupEvery = atoi(next());
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); die("bad option", __LINE__); }
  }
  const bool ref = (o.mode == "ref");
  if (ref) { o.gpus = 1; o.gmax = 0; if (g_R > 1) die("reference mode runs on one node", __LINE__); }
  if (o.gmax > 12) die("--gmax is at most 12", __LINE__);
  if (o.gmax < 0 || o.fixedSectors < 0 ||
      (o.fixedSectors && ((o.fixedSectors & (o.fixedSectors - 1)) || o.fixedSectors > (1 << o.gmax))))
    die("--fixed-sectors must be zero or a power of two within --gmax", __LINE__);
  if (o.policySeed < -1) die("--policy-seed must be -1 or nonnegative", __LINE__);
  if ((o.fixedSectors || o.policySeed >= 0) &&
      (o.mode != "engine" || o.fixedRows || o.window != "ann" || !o.resume.empty() || o.pilot || o.stream))
    die("count experiment requires engine/ann without fixed-rows, resume, pilot or stream", __LINE__);
  if (o.weight > 0) o.frame = 0;      // weight truncation needs the true strings

  Problem P = buildProblem(o);
  if (o.obsAll) {                // one term per qubit: the size of the problem grows with the number of qubits
    P.observable.clear();
    for (int q = 0; q < P.L.n; q++) singleZ(P, q);
  }
  g_L = P.L;
  int deferred = 0;
  g_ops = compileForEngine(P, o.frame != 0, &deferred);
  const int M = (int)g_ops.size(), words = g_L.words();
  int nBranch = 0, nMap = 0, nNoise = 0;
  for (auto& p : g_ops) { if (p.type == OP_ROT && p.branches) nBranch++; else if (p.type == OP_NOISE) nNoise++; else nMap++; }
  if (g_rank == 0)
    printf("circuit: %s; %d qubits; %zu gates -> %d operations on the data (%d branching rotations, %d Clifford, %d noise); "
           "%d Clifford gates deferred by the frame\n", P.name.c_str(), g_L.n, P.ops.size(), M, nBranch, nMap, nNoise, deferred);
  if (o.obsAll && g_rank == 0) printf("observable: sum of Z over the %d qubits (%zu terms)\n", g_L.n, P.observable.size());
  if (o.mode == "cupp") return runCuppDistributed(o, P);

  int nd = 0;
  CUDA(cudaGetDeviceCount(&nd));
  if (o.gpus > nd) die("fewer GPUs visible than requested", __LINE__);
  CUDA(cudaSetDevice(0));
  g_pool.cap = o.chunk; g_pool.pinned = (o.pinned != 0);

  std::vector<Worker> workers(o.gpus);
  for (int t = 0; t < o.gpus; t++) workers[t].init(t, o);
  CUDA(cudaSetDevice(0));
  if (g_rank == 0) {
    printf("mode=%s nodes=%d gpus/node=%d coef=%g weight=%d truncEvery=%d GPU capacity=%ld terms/buffer bucket target=%ld gmax=%d window=%s host=%s\n",
           o.mode.c_str(), g_R, o.gpus, o.coef, o.weight, o.truncEvery, workers[0].capTerms, o.target, o.gmax, o.window.c_str(),
           o.pinned ? "pinned chunks" : "pageable chunks");
    fflush(stdout);
  }
  if (g_rank == 0 && (o.fixedSectors || o.policySeed >= 0 || o.policyAudit))
    printf("count configuration: fixed_sectors=%d policy_seed=%ld bootstrap_workers=%d audit=%d\n",
           o.fixedSectors, o.policySeed, o.gpus * g_R, o.policyAudit);

  Shared sh;
  std::mt19937_64 rng(o.seed);
  std::vector<std::atomic<long>> popAt(o.profile.empty() ? 0 : M);
  std::vector<int> windowStarts;
  if (!o.profile.empty()) { for (auto& x : popAt) x = 0; sh.popAt = popAt.data(); }

  // Parities conserved by every branching rotation of the whole run, as masks for strings at
  // position 0. With a single-term observable every term must keep the observable's values.
  std::vector<BV> conserved;
  if (o.verify && P.observable.size() == 1) {
    Span all; all.L = g_L;
    std::vector<int> maps;
    for (int k = 0; k < M; k++) {
      const Op& op = g_ops[k];
      if (op.type == OP_ROT && op.branches) {
        BV v = op.gen;
        for (int j = (int)maps.size() - 1; j >= 0; --j) mapVec(g_ops[maps[j]], v.data(), g_L);
        all.add(v);
      } else if (changesBits(op)) maps.push_back(k);
    }
    conserved = all.annihilator();
    if (conserved.size() > 8) conserved.resize(8);
    sh.checkExpected = 0;
    for (size_t j = 0; j < conserved.size(); j++)
      sh.checkExpected |= dotParity(P.observable[0].first.data(), conserved[j].data(), words) << j;
  }
  int conservedAt = 0;      // position the masks in `conserved` currently refer to
  auto setCheckRows = [&](int position) {
    for (; conservedAt < position; conservedAt++)
      for (auto& r : conserved) mapMask(g_ops[conservedAt], r.data(), g_L);
    sh.checkG = (int)conserved.size();
    sh.checkRows.assign((size_t)std::max(sh.checkG, 1) * words, 0);
    for (int j = 0; j < sh.checkG; j++) std::copy(conserved[j].begin(), conserved[j].end(), sh.checkRows.begin() + (size_t)j * words);
  };

  // Initial pool: the observable, one bucket on node 0.
  std::vector<Bucket> pool(1);
  if (g_rank == 0)
    for (auto& t : P.observable) appendHost(pool[0], t.first.data(), &t.second, 1);
  int pos = 0, firstWindow = 0;
  Window w;
  bool haveWindow = false;

  // Checkpoint: every node writes its buckets and the window they were partitioned for.
  auto ckptName = [&](const std::string& base, int r) { return base + ".r" + std::to_string(r); };
  auto writeCheckpoint = [&](int position, int windowIndex, const Window& nw) {
    FILE* f = fopen(ckptName(o.ckptFile, g_rank).c_str(), "wb");
    if (!f) die("cannot open checkpoint for writing", __LINE__);
    int32_t hdr[6] = {0x50504B45, words, position, M, windowIndex, g_R};
    int wh[6]; std::vector<uint64_t> wp;
    packWindow(nw, wh, wp);
    int64_t wl = (int64_t)wp.size(), nbk = (int64_t)pool.size(), total = 0;
    fwrite(hdr, sizeof(int32_t), 6, f); fwrite(wh, sizeof(int), 6, f); fwrite(&wl, sizeof(int64_t), 1, f);
    fwrite(wp.data(), sizeof(uint64_t), wp.size(), f); fwrite(&nbk, sizeof(int64_t), 1, f);
    for (auto& b : pool) {
      int64_t n = b.n; total += n;
      fwrite(&n, sizeof(int64_t), 1, f);
      for (auto& k : b.ch) { fwrite(k.xz, sizeof(uint64_t), (size_t)k.n * words, f); }
      for (auto& k : b.ch) { fwrite(k.cf, sizeof(double), (size_t)k.n, f); }
    }
    fclose(f);
    long tl = total, tg = 0;
    MPI_Allreduce(&tl, &tg, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) printf("checkpoint written at operation %d: %ld terms on %d node(s)\n", position, tg, g_R);
  };
  if (!o.resume.empty()) {
    // Same node count: every node reloads its own buckets and continues the stored window.
    // One node: all files are merged into one bucket and the window is planned again.
    int files = 1;
    { FILE* f0 = fopen(ckptName(o.resume, 0).c_str(), "rb");
      if (!f0) die("cannot open checkpoint for reading", __LINE__);
      int32_t h[6]; if (fread(h, sizeof(int32_t), 6, f0) != 6) die("bad checkpoint", __LINE__);
      files = h[5]; fclose(f0); }
    if (g_R != 1 && g_R != files) die("resume on one node or on the node count of the checkpoint", __LINE__);
    const bool merge = (g_R == 1);
    for (auto& b : pool) releaseBucket(b);
    pool.assign(1, Bucket());
    long total = 0;
    std::vector<uint64_t> bx; std::vector<double> bc;
    for (int r = (g_R == 1 ? 0 : g_rank); r < (g_R == 1 ? files : g_rank + 1); r++) {
      FILE* f = fopen(ckptName(o.resume, r).c_str(), "rb");
      if (!f) die("cannot open checkpoint for reading", __LINE__);
      int32_t hdr[6]; int wh[6]; int64_t wl = 0, nbk = 0;
      if (fread(hdr, sizeof(int32_t), 6, f) != 6 || hdr[0] != 0x50504B45 || hdr[1] != words || hdr[3] != M) die("checkpoint does not match this circuit", __LINE__);
      if (fread(wh, sizeof(int), 6, f) != 6 || fread(&wl, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
      std::vector<uint64_t> wp((size_t)wl);
      if (fread(wp.data(), sizeof(uint64_t), wp.size(), f) != wp.size() || fread(&nbk, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
      pos = hdr[2]; firstWindow = hdr[4] + 1;
      if (!merge) { w = unpackWindow(wh, wp); haveWindow = true; pool.assign((size_t)nbk, Bucket()); }
      for (int64_t i = 0; i < nbk; i++) {
        int64_t n = 0;
        if (fread(&n, sizeof(int64_t), 1, f) != 1) die("bad checkpoint", __LINE__);
        bx.resize((size_t)n * words); bc.resize((size_t)n);
        if (fread(bx.data(), sizeof(uint64_t), bx.size(), f) != bx.size() || fread(bc.data(), sizeof(double), bc.size(), f) != bc.size()) die("bad checkpoint", __LINE__);
        appendHost(pool[merge ? 0 : (size_t)i], bx.data(), bc.data(), (long)n);
        total += n;
      }
      fclose(f);
    }
    long tg = 0;
    MPI_Allreduce(&total, &tg, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) printf("resumed at operation %d with %ld terms on %d node(s) (checkpoint was written by %d)\n", pos, tg, g_R, files);
  }

  // Host chunks prepared ahead of their use never take more than half of the node's memory.
  const long hostChunkLimit = (long)(0.5 * (double)sysconf(_SC_PHYS_PAGES) * (double)sysconf(_SC_PAGE_SIZE) / ((double)o.chunk * (words * 8 + 8)));
  double tPrealloc = 0;
  long chunksAhead = 0;
  // One GPU of one node: no other GPU could take a bucket, so buckets only keep the tasks at the target size.
  // While the operator is below that size it is one bucket and stays on the GPU between windows.
  const bool oneGpu = (o.gpus == 1 && g_R == 1 && !o.pilot);
  double tPlan = 0, growth = 2.0, tExchange = 0, bytesMoved = 0, worstCoset = 0, tPilot = 0, pilotGrowth = -1;
  long pilotEnds = 0;
  long maxBoundary = 0, maxInside = 0, movedBuckets = 0;
  int windows = 0, maxWindowOps = 0, minAnn = 2 * g_L.n;
  // Where the wall time goes on this node: its workers running a window, the wait for the slowest
  // node after it, and the part of the run with a single bucket (one GPU of one node works).
  double tRun = 0, tSync = 0, tSingle = 0, lastRunMax = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double epochBegin = epochNow();
  const double tBegin = now();

  std::vector<uint64_t> noSample;
  if (!haveWindow) w = planShared(pos, 0, noSample, o, rng);   // the pool is one bucket at the start
  int prevBreaks = 0;                                          // --fixed-rows: the last window ended with a changed row
  CountPolicy countPolicy(o.fixedSectors, o.gpus * g_R);
  for (int wi = firstWindow; ; wi++) {
    long NinLocal = 0, Nin = 0;
    for (auto& b : pool) NinLocal += b.n;
    const long onGpuTerms = (oneGpu && workers[0].onGpu) ? workers[0].numTerms() : 0;
    NinLocal += onGpuTerms;
    MPI_Allreduce(&NinLocal, &Nin, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    maxBoundary = std::max(maxBoundary, Nin);
    long trIn[2] = {0, 0}, trLargest = 0;     // --trace: non-empty buckets and the largest one at the window start
    if (o.trace) {
      long loc2[2] = {0, 0}, lmax = 0;
      for (auto& b : pool) if (b.n > 0) { loc2[0]++; lmax = std::max<long>(lmax, b.n); }
      if (onGpuTerms > 0) { loc2[0]++; lmax = std::max<long>(lmax, onGpuTerms); }
      MPI_Allreduce(loc2, trIn, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&lmax, &trLargest, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    }

    // Where should this window end? Its rows stay valid for any shorter window. With several buckets,
    // every node runs one of its buckets ahead and the window ends at the truncation after which those
    // pilots are smallest (in the second half of the window). Otherwise at the last truncation.
    if (!ref && w.e - w.b > 1) {
      const double tq = now();
      int newEnd = o.snap ? snapEnd(w) : w.e;
      if (o.pilot && w.g > 0 && Nin >= o.target) {
        const int len = w.e - w.b;
        std::vector<double> after(len, 0.0), tot(len, 0.0);
        std::vector<const Bucket*> cand;
        for (auto& b : pool) if (b.n > 0) cand.push_back(&b);
        int done = len;
        double scale = 0;
        if (!cand.empty()) {
          std::sort(cand.begin(), cand.end(), [](const Bucket* a, const Bucket* b) { return a->n < b->n; });
          const Bucket* pb = cand[cand.size() / 2];
          done = workers[0].pilot(w, *pb, o.pilotRows, after, scale);
          CUDA(cudaSetDevice(0));
        }
        int doneMin = len; double scaleTot = 0;
        MPI_Allreduce(&done, &doneMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(after.data(), tot.data(), len, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&scale, &scaleTot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (scaleTot > 0 && doneMin > len / 2) {
          double best = 1e300; int bestEnd = -1;
          for (int i = len / 2; i < doneMin; i++) {
            const int k = w.b + i;
            const bool candidate = g_ops[k].trunc || (k + 1 == w.e && doneMin == len);
            if (candidate && tot[i] < best) { best = tot[i]; bestEnd = k + 1; }
          }
          if (bestEnd > 0) { newEnd = bestEnd; pilotEnds++; pilotGrowth = best / scaleTot; }
        }
      }
      if (newEnd < w.e) w.lastBreaks = 0;
      w.e = newEnd;
      tPilot += now() - tq;
    }
    const bool last = (w.e >= M);

    // Plan the next window: its rows are scored on a sample of the current terms, moved through the
    // bit-changing operations of this window (branching is ignored for the estimate).
    Window nextW;
    int gWanted = 0;
    int adaptiveWanted = 0;
    std::vector<uint64_t> sample;
    // With several GPUs the rows are chosen while the workers run this window: they need them only to partition
    // their output. The sample is taken before, while the terms are still in the pool.
    const bool planLater = !last && !ref && !oneGpu && o.overlapPlan && !o.stream;
    auto planNext = [&] {
      const double tp = now();
      for (int k = w.b; k < w.e; k++)
        if (changesBits(g_ops[k]))
          for (size_t i = 0; i < sample.size() / words; i++) mapVec(g_ops[k], &sample[i * words], g_L);
      nextW = planShared(w.e, gWanted, sample, o, rng);
      tPlan += now() - tp;
    };
    if (!last) {
      const double tp = now();
      // Terms expected at the end of this window: from the pilot if there was one, else from the last ratio.
      if (!ref) {
        const double est = Nin * (pilotGrowth > 0 ? pilotGrowth * 1.3 : growth);
        while (gWanted < o.gmax && est / (double)(1L << gWanted) > (double)o.target) gWanted++;
        // Enough buckets that every GPU has several tasks per window, as long as buckets stay reasonably large
        // (not on one GPU: there is no other GPU to give a bucket to).
        // A bucket costs a fixed time per window (one library call per operation, however few terms): several
        // buckets per GPU pay off only when the windows are long.
        const double tasks = (double)(lastRunMax >= o.longWindow ? o.perGpu : 1) * o.gpus * g_R;
        while (!oneGpu && gWanted < o.gmax && (double)(1L << gWanted) < tasks && est / (double)(2L << gWanted) >= (double)o.minBucket) gWanted++;
        // The first windows grow fastest and the estimate lags behind: give every node a bucket early, so
        // that no node waits while one GPU carries the whole operator. Inside a node the bucket is shared
        // out as it grows (shareAbove).
        // By the start size (minBucketEarly) a problem that stays small is split over all nodes too; by the size
        // expected at the end of the window (minBucketEarlyEnd) only a window that has work for them is.
        if (o.minBucketEarlyEnd > 0)
          while (gWanted < o.gmax && (1L << gWanted) < (long)g_R * o.earlyPerNode && est * growth / (double)(2L << gWanted) >= (double)o.minBucketEarlyEnd) gWanted++;
        else
          while (gWanted < o.gmax && (1L << gWanted) < (long)g_R * o.earlyPerNode && est / (double)(2L << gWanted) >= (double)o.minBucketEarly) gWanted++;
      }
      adaptiveWanted = gWanted;
      gWanted = countPolicy.select(gWanted);
      pilotGrowth = -1;
      if (gWanted > 0) {
        const long stride = std::max<long>(1, Nin / std::max<long>(1, o.sample));   // o.sample terms over all nodes: every node sends its share
        if (onGpuTerms > 0) { workers[0].sampleOnGpu(sample, stride); CUDA(cudaSetDevice(0)); }
        for (auto& b : pool)
          for (auto& k : b.ch)
            for (long i = 0; i < k.n; i += stride) sample.insert(sample.end(), k.xz + (size_t)i * words, k.xz + (size_t)(i + 1) * words);
      }
      tPlan += now() - tp;
      if (!planLater) planNext();
    }
    windows++;
    windowStarts.push_back(w.b);
    maxWindowOps = std::max(maxWindowOps, w.e - w.b);
    minAnn = std::min(minAnn, w.annDim);
    worstCoset = std::max(worstCoset, w.cosetFrac);
    if (!conserved.empty()) setCheckRows(w.b);

    if (!planLater) sh.nextB.assign(last ? 1 : ((size_t)1 << nextW.g), Bucket());
    sh.planReady = !planLater;
    sh.nout = 0;
    sh.peakSum = 0;
    sh.canStay = false; sh.stayed = false;
    sh.nextRows = (last || planLater) ? 0 : nextW.g; sh.oneBucket = o.target;
    sh.splitEarly = oneGpu && w.g == 0;
    if (oneGpu) {
      int inputs = workers[0].onGpu ? 1 : 0;
      for (const auto& b : pool) inputs += b.n > 0 ? 1 : 0;
      sh.canStay = !last && inputs == 1;
      if (workers[0].onGpu) { sh.queue.push_back(Work()); sh.queue.back().b.n = workers[0].numTerms(); sh.queue.back().startOp = w.b; sh.queue.back().onGpu = true;
                              sh.queue.back().pendingTrunc = (prevBreaks && w.b > 0) ? (g_ops[w.b - 1].trunc ? 1 : 2) : 0; }
    }
    std::sort(pool.begin(), pool.end(), [](const Bucket& a, const Bucket& b) { return a.n > b.n; });
    for (auto& b : pool)
      if (b.n > 0) { sh.queue.push_back(Work()); sh.queue.back().b.ch.swap(b.ch); sh.queue.back().b.n = b.n; sh.queue.back().startOp = w.b; b.n = 0;
                     sh.queue.back().pendingTrunc = (prevBreaks && w.b > 0) ? (g_ops[w.b - 1].trunc ? 1 : 2) : 0; }
      else releaseBucket(b);
    sh.active = 0;
    std::vector<std::thread> th;
    sh.finished = 0;
    const double tr0 = now();
    for (int t = 0; t < o.gpus; t++) th.emplace_back([&, t] { workers[t].run(w, last ? nullptr : &nextW, sh, !ref); sh.finished++; });
    if (planLater) {
      planNext();
      sh.nextB.assign((size_t)1 << nextW.g, Bucket());
      sh.nextRows = nextW.g;
      { std::lock_guard<std::mutex> lk(sh.planM); sh.planReady = true; }
      sh.planCv.notify_all();
    }
    if (g_R > 1 && !last && o.stream) streamExchange(sh.nextB, sh.bm, sh.finished, o.gpus, tExchange, bytesMoved);
    else if (!last && g_R > 1 && g_xNewChunks > 0) {
      // The last exchange had to allocate chunks for the arriving terms while all GPUs of the node waited: the
      // operator is growing. While the GPUs work on this window, this thread prepares what the next exchange is
      // expected to need on top: as many chunks again, times the growth of the last window (at most twice).
      const long want = std::min(g_pool.total() + (long)((double)g_xNewChunks * std::min(2.0, std::max(1.0, growth / 1.3))), hostChunkLimit);
      while (sh.finished.load() < o.gpus) {
        if (g_pool.total() < want) { const double ta = now(); g_pool.put(g_pool.fresh()); tPrealloc += now() - ta; chunksAhead += g_pool.pinned ? ChunkPool::kSlab : 1; }
        else std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    }
    for (auto& x : th) x.join();
    if (sh.stayed && !last && nextW.g > 0) {
      // The result stayed on the GPU: the next window is one bucket, and its rows remain available for splitting.
      for (int j = 0; j < nextW.g; j++) nextW.spare.emplace_back(nextW.rows.begin() + (size_t)j * words, nextW.rows.begin() + (size_t)(j + 1) * words);
      nextW.g = 0; nextW.rows.clear();
    }
    const double tr1 = now();
    tRun += tr1 - tr0;

    long NoutLocal = sh.nout.load(), Nout = 0;
    MPI_Allreduce(&NoutLocal, &Nout, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    { long pl = sh.peakSum.load(), pg = 0; MPI_Allreduce(&pl, &pg, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); maxInside = std::max(maxInside, pg); }
    tSync += now() - tr1;
    { double tm = tr1 - tr0; lastRunMax = tm; MPI_Allreduce(&tm, &lastRunMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); }
    if (w.g == 0) tSingle += lastRunMax;
    if (o.trace) {
      long sl[2] = {sh.splits.load(), sh.runs.load()}, sg[2];
      MPI_Allreduce(sl, sg, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      int rot = 0;
      for (int k = w.b; k < w.e; k++) rot += (g_ops[k].type == OP_ROT && g_ops[k].branches) ? 1 : 0;
      if (g_rank == 0) {
        printf("trace window=%d ops=%d..%d branching_rotations=%d generator_rank=%d valid_rows=%d rows=%d sectors=%ld largest_sector_in=%ld terms_in=%ld terms_out=%ld next_rows=%d splits_so_far=%ld sector_runs_so_far=%ld run=%.3f s\n",
               wi, w.b, w.e, rot, 2 * g_L.n - w.annDim, w.annDim, w.g, trIn[0], trLargest, Nin, Nout, last ? -1 : nextW.g, sg[0], sg[1], lastRunMax);
        fflush(stdout);
      }
    }
    if (Nin > 0 && Nout > 0) growth = std::max(1.0, std::min(o.growthCap, (double)Nout / (double)Nin * 1.3));
    if (o.policyAudit && g_rank == 0) {
      Span rowSpan; rowSpan.L = g_L;
      uint64_t rowHash = 14695981039346656037ULL;
      for (int j = 0; j < w.g; ++j) {
        BV row(w.rows.begin() + (size_t)j * words, w.rows.begin() + (size_t)(j + 1) * words);
        rowSpan.add(row);
        for (uint64_t x : row) { rowHash ^= x; rowHash *= 1099511628211ULL; }
      }
      printf("count_policy window=%d rows=%d labels=%ld row_rank=%d row_hash=%016llx adaptive_next_rows=%d requested_next_rows=%d next_fixed_active=%d\n",
             wi, w.g, 1L << w.g, rowSpan.dim(), (unsigned long long)rowHash,
             last ? -1 : adaptiveWanted, last ? -1 : gWanted, countPolicy.active ? 1 : 0);
      fflush(stdout);
    }
    if (last) break;
    if (!o.stream) exchangeBuckets(sh.nextB, tExchange, bytesMoved);
    pool.swap(sh.nextB);
    if (nextW.g > 0) movedBuckets++;
    pos = w.e;
    if (wi == o.ckptWindow && !o.ckptFile.empty()) {
      if (oneGpu) workers[0].toHostFromGpu(pool[0]);
      writeCheckpoint(pos, wi, nextW);
      if (o.stopAtCkpt) { if (g_rank == 0) printf("stopped at checkpoint\n"); MPI_Finalize(); return 0; }
    }
    prevBreaks = w.lastBreaks;
    w = nextW;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = now() - tBegin;
  if (g_rank == 0) printf("solve_epoch=%.3f..%.3f\n", epochBegin, epochNow());

  double loc[10] = {0, 0, 0, 0, 0, tExchange, bytesMoved, sh.expec, 0, 0}, sum[10];
  for (auto& x : workers) { loc[0] += x.tOps; loc[1] += x.tPart; loc[2] += x.tH2D; loc[3] += x.tD2H; loc[4] += x.tIdle; loc[8] += x.tVerify; loc[9] += x.tPlanWait; }
  MPI_Allreduce(loc, sum, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  long lc[7] = {sh.runs.load(), sh.splits.load(), sh.hostSplits.load(), sh.peak.load(), sh.verified.load(), sh.violations.load(), g_pool.allocated}, sc[7], mc[7];
  MPI_Allreduce(lc, sc, 7, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(lc, mc, 7, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  double twl[2] = {tRun, tSync}, tw[2];
  MPI_Allreduce(twl, tw, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  long tll[2] = {sh.shares.load(), sh.earlyDedups.load()}, tl[2];
  MPI_Allreduce(tll, tl, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  double xl[7] = {g_xCounts, g_xPost, g_xAlloc, g_xWait, (double)g_pool.allocNs.load() * 1e-9, tPrealloc, (double)chunksAhead}, xs[7];
  MPI_Allreduce(xl, xs, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  long nfl[2] = {0, 0}, nf[2] = {0, 0};
  for (auto& x : workers) { nfl[0] += x.noiseFusion.calls; nfl[1] += x.noiseFusion.operations; }
  MPI_Allreduce(nfl, nf, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) {
    printf("expectation=%.12g\n", sum[7]);
    printf("windows=%d (longest %d operations, smallest row space %d of %d, largest coset share %.4f) partitioned boundaries=%ld\n",
           windows, maxWindowOps, minAnn, 2 * g_L.n, worstCoset, movedBuckets);
    printf("max_terms_at_window_boundary=%ld largest_bucket_on_a_gpu=%ld bucket_runs=%ld splits_on_gpu=%ld splits_on_host=%ld\n",
           maxBoundary, mc[3], sc[0], sc[1], sc[2]);
    // what an implementation that keeps the whole operator at once would hold: every sector at its largest
    printf("max_terms_inside_a_window=%ld (sum over the sector runs of one window of the largest count each reached)\n", maxInside);
    if (!conserved.empty())
      printf("integrity: %ld term checks against %zu conserved parities, %ld violations (%.3f s summed over workers)\n", sc[4], conserved.size(), sc[5], sum[8]);
    printf("wall=%.3f s; summed over %d GPU workers: operations=%.3f partition=%.3f host->gpu=%.3f gpu->host=%.3f idle=%.3f; planning=%.3f; pilot=%.3f (%ld window ends chosen)\n",
           wall, o.gpus * g_R, sum[0], sum[1], sum[2], sum[3], sum[4], tPlan, tPilot, pilotEnds);
    if (!oneGpu && o.overlapPlan && !ref)
      printf("rows of the next window chosen while the window runs: %.3f s on the main thread of node 0; workers waited %.3f s for them (summed)\n", tPlan, sum[9]);
    printf("host memory in chunks: %.2f GiB on the busiest node, %.2f GiB over all nodes\n",
           mc[6] * (double)o.chunk * (words * 8 + 8) / 1073741824.0, sc[6] * (double)o.chunk * (words * 8 + 8) / 1073741824.0);
    if (oneGpu) printf("one GPU: the operator stayed on the GPU after %ld of %d windows\n", workers[0].windowsOnGpu, windows);
    if (!o.profile.empty() && g_R == 1) {
      FILE* f = fopen(o.profile.c_str(), "w");
      if (f) {
        fprintf(f, "op,branching,trunc,window_start,terms_before\n");
        size_t wi2 = 0;
        for (int k = 0; k < M; k++) {
          int ws = 0;
          while (wi2 < windowStarts.size() && windowStarts[wi2] < k) wi2++;
          if (wi2 < windowStarts.size() && windowStarts[wi2] == k) ws = 1;
          fprintf(f, "%d,%d,%d,%d,%ld\n", k, (int)(g_ops[k].type == OP_ROT && g_ops[k].branches), (int)g_ops[k].trunc, ws, popAt[k].load());
        }
        fclose(f);
      }
    }
    printf("wall on a node, averaged over nodes: windows running=%.3f, waiting for the slowest node=%.3f; windows with a single bucket=%.3f; buckets shared out inside a node=%ld\n",
           tw[0] / g_R, tw[1] / g_R, tSingle, tl[0]);
    if (nf[0] > 0) printf("noise channels applied in groups: %ld operations in %ld calls (summed over buckets)\n", nf[1], nf[0]);
    if (g_R > 1)
      printf("between nodes: %.3f GB sent, %.3f s per node on average with all GPUs idle (%s)\n",
             sum[6] / 1e9, sum[5] / g_R, o.stream ? "exchange overlapped with the windows; this is the wait for the last chunks" : "exchange after each window");
    if (g_R > 1 && !o.stream)
      printf("exchange per node on average: counts %.3f s, posting %.3f s (of it new host chunks %.3f s), transfer %.3f s\n",
             xs[0] / g_R, xs[1] / g_R, xs[2] / g_R, xs[3] / g_R);
    printf("new host chunks: %.3f s per node on average, summed over the threads of a node (%ld chunks on the busiest node); of it %.3f s for %.0f chunks prepared while the GPUs worked\n",
           xs[4] / g_R, mc[6], xs[5] / g_R, xs[6] / g_R);
  }
  MPI_Finalize();
  return 0;
}
