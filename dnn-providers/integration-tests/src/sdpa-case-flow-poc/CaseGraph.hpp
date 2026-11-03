// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>

namespace sdpa_case_flow
{

// One case: model axis id -> model value id. A child axis whose parent is not active holds "-".
using Row = std::map<std::string, std::string>;

// Thrown when a row cannot become a graph: an unknown value id, a combination the
// class-to-value table does not cover, or a malformed override token.
class CaseRealisationError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Thrown when the class-to-value file is malformed or lacks a table the driver needs.
class ClassValuesError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

enum class DescaleForm
{
    BY_VALUE,
    DEVICE,
};

struct ScaleNumber
{
    bool inverseSqrtHeadDim = false;
    double value = 0.0;
};

// Head counts that replace a power-of-two query head count for ALiBi tests.
struct AlibiHeads
{
    int64_t hq = 0;
    int64_t mha = 0;
    int64_t gqa = 0;
    int64_t mqa = 0;
    int64_t differK = 0;
    int64_t differV = 0;
};

struct SequenceLengthEntry
{
    std::string seqQ;
    std::string seqKv;
    std::string relation;
    int64_t sq = 0;
    int64_t skv = 0;
};

// The concrete values of the model's classes, as written by the case generator.
struct ClassValues
{
    std::map<std::string, int64_t> batchSize;
    std::map<std::string, int64_t> qHeads;
    std::vector<SequenceLengthEntry> sequenceLengths;
    std::map<std::string, int64_t> headDim;
    // dtype_q value -> format of a K or V whose relation to Q is "differs".
    std::map<std::string, std::string> differsFormat;
    // Base arch -> (format -> the format used on that arch).
    std::map<std::string, std::map<std::string, std::string>> archFormat;
    // Physical order, outermost first, as indices into the logical dims [B, H, S, D].
    std::map<std::string, std::vector<size_t>> dimOrder;
    // Representatives of each class, primary first. A case uses the primary.
    std::map<std::string, std::vector<ScaleNumber>> scaleValue;
    std::map<std::string, double> dropoutProbability;
    int64_t dropoutSeed = 0;
    int64_t dropoutOffset = 0;
    std::map<std::string, double> dropoutMaskScale;
    // dropout_mask_pattern value -> mask contents, repeated in element order.
    std::map<std::string, std::vector<double>> dropoutMaskCycle;
    // Tokens per page; std::nullopt means one page per sequence (S_kv tokens).
    std::map<std::string, std::optional<int64_t>> pageSize;
    // Keys descale_q, descale_k, descale_v and other.
    std::map<std::string, double> quantisationFactors;
    // Granularity (per_head, finer) -> descale name -> factors, repeated in element order.
    std::map<std::string, std::map<std::string, std::vector<double>>> descaleLists;
    // batch_packing value -> ragged tensors, as letters Q, K, V, O.
    std::map<std::string, std::set<std::string>> raggedTensors;
    // paged_kv value -> paged sides, as letters K, V.
    std::map<std::string, std::set<std::string>> pagedSides;
    // page_size_k_vs_v value -> V page size as a multiple of K's.
    std::map<std::string, int64_t> vPageMultiplier;
    // page_table_sharing value -> number of page-table tensors (1 or 2).
    std::map<std::string, int64_t> pageTables;
    // page_table_storage value -> whether the table carries a ragged offset.
    std::map<std::string, bool> packedPageTable;
    // Power-of-two query head count -> the counts that replace it.
    std::map<int64_t, AlibiHeads> alibiHeads;
    // fp8 mma_core_mode when Q is not fp8, and per-arch substitutions.
    std::string mmaFp8Otherwise;
    std::map<std::string, std::map<std::string, std::string>> mmaFp8ByArch;
};

// Parse the generator's class-to-value JSON. Throws ClassValuesError.
ClassValues parseClassValues(const std::string& jsonText);

// Concrete tensor extents that replace the class-to-value sizes of a row.
struct SizeOverride
{
    int64_t b = 0;
    int64_t hq = 0;
    int64_t hk = 0;
    int64_t hv = 0;
    int64_t sq = 0;
    int64_t skv = 0;
    int64_t dqk = 0;
    int64_t dv = 0;
};

// Parse "B2 Hq8 Hk1 Hv1 Sq64 Skv1 Dqk256 Dv128". Throws CaseRealisationError.
SizeOverride parseSizeOverride(const std::string& text);

struct CaseContext
{
    const ClassValues& values;
    DescaleForm descaleForm;
    // Base arch of the device (for example gfx942), or "none".
    std::string arch;
    // When set, these extents replace the class-to-value sizes of the row.
    std::optional<SizeOverride> sizes = std::nullopt;
};

struct BuiltCase
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    // Concrete numbers and formats the row turned into, for the result record.
    std::string realised;
    // Values a tensor's contents must hold, keyed by tensor name. Element i of the named
    // tensor is set to values[i % values.size()] in place of the random input fill.
    std::map<std::string, std::vector<float>> floatContents;
    // Whole contents of integer tensors, keyed by tensor name, in packed element order.
    std::map<std::string, std::vector<int64_t>> indexContents;
};

// The axis ids of the maintained forward SDPA model, in its column order.
const std::vector<std::string>& modelAxisIds();

// True for an axis whose value is decided by tensor contents or pointers at execute time.
bool isExecuteLevelAxis(const std::string& axis);

// True for an execute-level axis that buildCaseGraph turns into BuiltCase::floatContents.
bool isRealisedContentAxis(const std::string& axis);

// The shared base case value of an axis, or "-" when the base omits the axis.
std::string sharedBaseValue(const std::string& axis);

// Build one forward SDPA graph from a row and its override tokens. Nothing is lowered,
// validated or sent to a backend here.
BuiltCase buildCaseGraph(const Row& row,
                         const std::vector<std::string>& overrideTokens,
                         const CaseContext& context);

} // namespace sdpa_case_flow
