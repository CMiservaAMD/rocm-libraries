// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Acceptance-only discovery driver for forward SDPA cases.
//
// Each input row becomes one forward SDPA graph through the hipDNN C++ frontend. The
// driver asks one pinned engine whether it accepts the graph and writes one JSON line
// per case. It never executes a graph. With --smoke-build it builds, but never runs,
// a plan for each accepted case.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hipdnn_backend.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/PluginPaths.hpp>

#include "CaseGraph.hpp"

namespace
{

using sdpa_case_flow::Row;
using Clock = std::chrono::steady_clock;

const char* const OUTCOME_FRONTEND_REJECTED = "frontend-rejected";
const char* const OUTCOME_NOT_APPLICABLE = "not-applicable";
const char* const OUTCOME_ACCEPTED = "accepted";
const char* const OUTCOME_ERROR = "error";
const char* const OUTCOME_NOT_PROBED = "not-probed";

// The backend's message when an engine ID is not among the engines that accept the graph
// (EngineDescriptor::finalize).
const char* const ENGINE_REFUSED_MARKER = "not in a valid range of engine IDs";

constexpr int EXIT_USAGE = 1;
constexpr int EXIT_ARCH_MISMATCH = 2;
constexpr int EXIT_ENGINE_NOT_LOADED = 3;
constexpr int EXIT_NO_CASE_ACCEPTED = 4;

constexpr size_t MAX_LOG_LINES_PER_CASE = 8;
constexpr size_t MAX_LOG_LINE_LENGTH = 400;

struct Options
{
    std::string input;
    std::string output;
    std::string classValues;
    std::string engine = "ASM_SDPA_ENGINE";
    std::vector<std::string> pluginPaths;
    std::string expectArch;
    int64_t repeat = 1;
    int64_t limit = -1;
    bool smokeBuild = false;
    bool otherEngines = false;
    bool allowNoAccepted = false;
    bool overwrite = false;
    std::string captureLog;
    sdpa_case_flow::DescaleForm descaleForm = sdpa_case_flow::DescaleForm::BY_VALUE;
    std::vector<std::pair<std::string, std::string>> meta;
};

void printUsage(std::ostream& os)
{
    os << "Usage: hipdnn_sdpa_case_flow_discovery --input CASES.csv --output RESULTS.jsonl "
          "--class-values class-to-value.json [options]\n"
          "  --class-values FILE      the generator's class-to-value file (concrete values of "
          "classes)\n"
          "  --engine NAME            engine to pin by name (default ASM_SDPA_ENGINE)\n"
          "  --plugin-path PATH       load only these engine plugins; repeat for more\n"
          "  --expect-arch GFX        stop before any query when HIP reports another arch\n"
          "  --repeat N               query each case N times (default 1)\n"
          "  --limit N                use only the first N cases\n"
          "  --smoke-build            build, never execute, a plan for each accepted case\n"
          "  --other-engines          for each refused case, list the engines that accept it\n"
          "  --descale-form FORM      by-value (default) or device, for descale/scale tensors\n"
          "  --capture-log LEVEL      attach hipDNN log lines (info|warn|error) to each case\n"
          "  --allow-no-accepted      exit 0 even when no case is accepted\n"
          "  --overwrite              replace an existing output file\n"
          "  --meta KEY=VALUE         copy KEY=VALUE into the run record; repeat for more\n";
}

[[noreturn]] void usageError(const std::string& message)
{
    std::cerr << "error: " << message << "\n";
    printUsage(std::cerr);
    std::exit(EXIT_USAGE);
}

int64_t parsePositive(const std::string& flag, const std::string& text)
{
    try
    {
        size_t used = 0;
        const auto value = std::stoll(text, &used);
        if(used == text.size() && value > 0)
        {
            return value;
        }
    }
    catch(const std::exception&)
    {
    }
    usageError(flag + " needs a positive integer");
}

Options parseArgs(int argc, char** argv)
{
    Options opts;
    const auto next = [&](int& i, const std::string& flag) -> std::string {
        if(i + 1 >= argc)
        {
            usageError(flag + " needs a value");
        }
        return argv[++i];
    };

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--input")
        {
            opts.input = next(i, arg);
        }
        else if(arg == "--output")
        {
            opts.output = next(i, arg);
        }
        else if(arg == "--class-values")
        {
            opts.classValues = next(i, arg);
        }
        else if(arg == "--engine")
        {
            opts.engine = next(i, arg);
        }
        else if(arg == "--plugin-path")
        {
            opts.pluginPaths.push_back(next(i, arg));
        }
        else if(arg == "--expect-arch")
        {
            opts.expectArch = next(i, arg);
        }
        else if(arg == "--repeat")
        {
            opts.repeat = parsePositive(arg, next(i, arg));
        }
        else if(arg == "--limit")
        {
            opts.limit = parsePositive(arg, next(i, arg));
        }
        else if(arg == "--smoke-build")
        {
            opts.smokeBuild = true;
        }
        else if(arg == "--other-engines")
        {
            opts.otherEngines = true;
        }
        else if(arg == "--allow-no-accepted")
        {
            opts.allowNoAccepted = true;
        }
        else if(arg == "--overwrite")
        {
            opts.overwrite = true;
        }
        else if(arg == "--descale-form")
        {
            const auto form = next(i, arg);
            if(form == "by-value")
            {
                opts.descaleForm = sdpa_case_flow::DescaleForm::BY_VALUE;
            }
            else if(form == "device")
            {
                opts.descaleForm = sdpa_case_flow::DescaleForm::DEVICE;
            }
            else
            {
                usageError("--descale-form needs by-value or device");
            }
        }
        else if(arg == "--capture-log")
        {
            opts.captureLog = next(i, arg);
            if(opts.captureLog != "info" && opts.captureLog != "warn" && opts.captureLog != "error")
            {
                usageError("--capture-log needs info, warn or error");
            }
        }
        else if(arg == "--meta")
        {
            const auto pair = next(i, arg);
            const auto eq = pair.find('=');
            if(eq == std::string::npos || eq == 0)
            {
                usageError("--meta needs KEY=VALUE");
            }
            opts.meta.emplace_back(pair.substr(0, eq), pair.substr(eq + 1));
        }
        else if(arg == "--help" || arg == "-h")
        {
            printUsage(std::cout);
            std::exit(0);
        }
        else
        {
            usageError("unknown argument '" + arg + "'");
        }
    }

    if(opts.input.empty() || opts.output.empty() || opts.classValues.empty())
    {
        usageError("--input, --output and --class-values are required");
    }
    if(std::filesystem::exists(opts.output) && !opts.overwrite)
    {
        usageError("output file exists; pass --overwrite to replace it");
    }
    return opts;
}

// ------------------------------------------------------------------ SHA-256 of the input file

class Sha256
{
public:
    void update(const unsigned char* data, size_t size)
    {
        for(size_t i = 0; i < size; ++i)
        {
            _block[_blockSize++] = data[i];
            if(_blockSize == 64)
            {
                transform();
                _bitCount += 512;
                _blockSize = 0;
            }
        }
    }

    std::string hexDigest()
    {
        const uint64_t totalBits = _bitCount + static_cast<uint64_t>(_blockSize) * 8;
        _block[_blockSize++] = 0x80;
        if(_blockSize > 56)
        {
            while(_blockSize < 64)
            {
                _block[_blockSize++] = 0;
            }
            transform();
            _blockSize = 0;
        }
        while(_blockSize < 56)
        {
            _block[_blockSize++] = 0;
        }
        for(int shift = 56; shift >= 0; shift -= 8)
        {
            _block[_blockSize++] = static_cast<unsigned char>((totalBits >> shift) & 0xffU);
        }
        transform();

        std::ostringstream os;
        for(const uint32_t word : _state)
        {
            os << std::hex << std::setw(8) << std::setfill('0') << word;
        }
        return os.str();
    }

private:
    std::array<uint32_t, 8> _state = {0x6a09e667U,
                                      0xbb67ae85U,
                                      0x3c6ef372U,
                                      0xa54ff53aU,
                                      0x510e527fU,
                                      0x9b05688cU,
                                      0x1f83d9abU,
                                      0x5be0cd19U};
    std::array<unsigned char, 64> _block{};
    size_t _blockSize = 0;
    uint64_t _bitCount = 0;

    static uint32_t rotr(uint32_t x, uint32_t n)
    {
        return (x >> n) | (x << (32U - n));
    }

    void transform()
    {
        static const std::array<uint32_t, 64> K
            = {0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
               0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
               0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
               0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
               0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
               0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
               0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
               0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
               0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
               0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
               0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

        std::array<uint32_t, 64> w{};
        for(size_t i = 0; i < 16; ++i)
        {
            w[i] = (static_cast<uint32_t>(_block[i * 4]) << 24)
                   | (static_cast<uint32_t>(_block[i * 4 + 1]) << 16)
                   | (static_cast<uint32_t>(_block[i * 4 + 2]) << 8)
                   | static_cast<uint32_t>(_block[i * 4 + 3]);
        }
        for(size_t i = 16; i < 64; ++i)
        {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = _state[0];
        uint32_t b = _state[1];
        uint32_t c = _state[2];
        uint32_t d = _state[3];
        uint32_t e = _state[4];
        uint32_t f = _state[5];
        uint32_t g = _state[6];
        uint32_t h = _state[7];
        for(size_t i = 0; i < 64; ++i)
        {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + K[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        _state[0] += a;
        _state[1] += b;
        _state[2] += c;
        _state[3] += d;
        _state[4] += e;
        _state[5] += f;
        _state[6] += g;
        _state[7] += h;
    }
};

// ------------------------------------------------------------------ JSON output

std::string jsonString(const std::string& text)
{
    std::string out = "\"";
    for(const char ch : text)
    {
        const auto uch = static_cast<unsigned char>(ch);
        switch(ch)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if(uch < 0x20)
            {
                std::ostringstream os;
                os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << static_cast<unsigned int>(uch);
                out += os.str();
            }
            else
            {
                out += ch;
            }
            break;
        }
    }
    return out + "\"";
}

std::string jsonNumber(double value)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(3) << value;
    return os.str();
}

std::string jsonStringArray(const std::vector<std::string>& items)
{
    std::string out = "[";
    for(size_t i = 0; i < items.size(); ++i)
    {
        out += (i == 0 ? "" : ",") + jsonString(items[i]);
    }
    return out + "]";
}

class JsonLine
{
public:
    JsonLine& str(const std::string& key, const std::string& value)
    {
        return raw(key, jsonString(value));
    }
    JsonLine& num(const std::string& key, double value)
    {
        return raw(key, jsonNumber(value));
    }
    JsonLine& integer(const std::string& key, int64_t value)
    {
        return raw(key, std::to_string(value));
    }
    JsonLine& boolean(const std::string& key, bool value)
    {
        return raw(key, value ? "true" : "false");
    }
    JsonLine& raw(const std::string& key, const std::string& jsonValue)
    {
        _text += (_text.empty() ? "{" : ",") + jsonString(key) + ":" + jsonValue;
        return *this;
    }
    std::string done() const
    {
        return _text + "}";
    }

private:
    std::string _text;
};

// ------------------------------------------------------------------ input

struct CaseInput
{
    Row row;
    std::string caseId;
    std::vector<std::string> overrides;
    std::vector<std::string> decidingAxes;
    bool needsExecution = false;
};

std::vector<std::string> split(const std::string& text, char separator)
{
    std::vector<std::string> parts;
    std::string part;
    std::stringstream stream(text);
    while(std::getline(stream, part, separator))
    {
        parts.push_back(part);
    }
    if(!text.empty() && text.back() == separator)
    {
        parts.emplace_back();
    }
    return parts;
}

std::string trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t\r");
    if(first == std::string::npos)
    {
        return "";
    }
    const auto last = text.find_last_not_of(" \t\r");
    return text.substr(first, last - first + 1);
}

std::vector<std::string> axesNamedIn(const std::string& cell)
{
    std::vector<std::string> axes;
    for(const auto& part : split(cell, '|'))
    {
        const auto eq = part.find('=');
        if(eq != std::string::npos && eq > 0)
        {
            axes.push_back(trim(part.substr(0, eq)));
        }
    }
    return axes;
}

[[noreturn]] void inputError(const std::string& message)
{
    std::cerr << "error: " << message << "\n";
    std::exit(EXIT_USAGE);
}

struct InputFile
{
    std::vector<std::string> header;
    std::vector<std::string> ignoredColumns;
    std::vector<CaseInput> cases;
    std::string sha256;
    size_t dataRows = 0;
};

InputFile readInput(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
    {
        inputError("cannot open input '" + path + "'");
    }
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());

    InputFile file;
    Sha256 sha;
    sha.update(reinterpret_cast<const unsigned char*>(content.data()), content.size());
    file.sha256 = sha.hexDigest();

    if(content.find('"') != std::string::npos)
    {
        inputError("quoted CSV fields are not supported");
    }

    std::vector<std::string> lines = split(content, '\n');
    while(!lines.empty() && trim(lines.back()).empty())
    {
        lines.pop_back();
    }
    if(lines.empty())
    {
        inputError("input has no header line");
    }

    for(const auto& column : split(lines[0], ','))
    {
        file.header.push_back(trim(column));
    }
    const std::set<std::string> headerSet(file.header.begin(), file.header.end());
    if(headerSet.size() != file.header.size())
    {
        inputError("input header repeats a column");
    }
    if(headerSet.count("case_id") == 0)
    {
        inputError("input header has no case_id column");
    }
    for(const auto& axis : sdpa_case_flow::modelAxisIds())
    {
        if(headerSet.count(axis) == 0)
        {
            inputError("input header has no column for model axis '" + axis + "'");
        }
    }

    const std::set<std::string> axisSet(sdpa_case_flow::modelAxisIds().begin(),
                                        sdpa_case_flow::modelAxisIds().end());
    static const std::set<std::string> USED_COLUMNS
        = {"case_id", "varied", "pair", "overrides", "needs_execution"};
    for(const auto& column : file.header)
    {
        if(axisSet.count(column) == 0 && USED_COLUMNS.count(column) == 0)
        {
            file.ignoredColumns.push_back(column);
        }
    }

    for(size_t lineNo = 1; lineNo < lines.size(); ++lineNo)
    {
        if(trim(lines[lineNo]).empty())
        {
            continue;
        }
        const auto fields = split(lines[lineNo], ',');
        if(fields.size() != file.header.size())
        {
            inputError("line " + std::to_string(lineNo + 1) + " has "
                       + std::to_string(fields.size()) + " fields; the header has "
                       + std::to_string(file.header.size()));
        }
        CaseInput input;
        for(size_t i = 0; i < fields.size(); ++i)
        {
            const auto& column = file.header[i];
            const auto field = trim(fields[i]);
            if(axisSet.count(column) != 0)
            {
                input.row[column] = field;
            }
            else if(column == "case_id")
            {
                input.caseId = field;
            }
            else if(column == "varied" || column == "pair")
            {
                for(const auto& axis : axesNamedIn(field))
                {
                    input.decidingAxes.push_back(axis);
                }
            }
            else if(column == "overrides")
            {
                for(const auto& token : split(field, ';'))
                {
                    if(!trim(token).empty())
                    {
                        input.overrides.push_back(trim(token));
                    }
                }
            }
            else if(column == "needs_execution")
            {
                input.needsExecution = field == "true" || field == "1" || field == "yes";
            }
        }
        file.cases.push_back(std::move(input));
    }
    file.dataRows = file.cases.size();
    return file;
}

struct ClassValuesFile
{
    sdpa_case_flow::ClassValues values;
    std::string sha256;
};

ClassValuesFile readClassValues(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
    {
        inputError("cannot open class-to-value file '" + path + "'");
    }
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());

    ClassValuesFile file;
    Sha256 sha;
    sha.update(reinterpret_cast<const unsigned char*>(content.data()), content.size());
    file.sha256 = sha.hexDigest();
    try
    {
        file.values = sdpa_case_flow::parseClassValues(content);
    }
    catch(const sdpa_case_flow::ClassValuesError& e)
    {
        inputError(std::string(e.what()) + " ('" + path + "')");
    }
    return file;
}

// ------------------------------------------------------------------ device and environment

struct DeviceInfo
{
    int deviceCount = 0;
    int device = -1;
    std::string arch = "none";
    std::string name;
    std::string hipError;
};

DeviceInfo queryDevice()
{
    DeviceInfo info;
    int count = 0;
    const hipError_t countStatus = hipGetDeviceCount(&count);
    if(countStatus != hipSuccess)
    {
        info.hipError = std::string("hipGetDeviceCount: ") + hipGetErrorString(countStatus);
        return info;
    }
    info.deviceCount = count;
    if(count <= 0)
    {
        return info;
    }
    int device = 0;
    const hipError_t deviceStatus = hipGetDevice(&device);
    if(deviceStatus != hipSuccess)
    {
        info.hipError = std::string("hipGetDevice: ") + hipGetErrorString(deviceStatus);
        return info;
    }
    info.device = device;
    hipDeviceProp_t props{};
    const hipError_t propStatus = hipGetDeviceProperties(&props, device);
    if(propStatus != hipSuccess)
    {
        info.hipError = std::string("hipGetDeviceProperties: ") + hipGetErrorString(propStatus);
        return info;
    }
    info.arch = props.gcnArchName;
    info.name = props.name;
    return info;
}

std::string baseArch(const std::string& arch)
{
    return arch.substr(0, arch.find(':'));
}

std::string envOr(const char* name, const std::string& fallback)
{
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value);
}

std::string hostName()
{
    std::array<char, 256> buffer{};
    if(gethostname(buffer.data(), buffer.size() - 1) != 0)
    {
        return "unknown";
    }
    return buffer.data();
}

// ------------------------------------------------------------------ hipDNN helpers

std::string engineName(hipdnnHandle_t handle, int64_t engineId)
{
    size_t length = 0;
    if(hipdnnGetEngineNameById_ext(handle, engineId, nullptr, &length) == HIPDNN_STATUS_SUCCESS
       && length > 0)
    {
        std::vector<char> buffer(length);
        if(hipdnnGetEngineNameById_ext(handle, engineId, buffer.data(), &length)
           == HIPDNN_STATUS_SUCCESS)
        {
            return buffer.data();
        }
    }
    return hipdnn_data_sdk::utilities::engineNameOrHex(engineId);
}

struct LogCapture
{
    std::mutex mutex;
    bool active = false;
    std::vector<std::string> lines;
};

void captureLogLine(hipdnnUserLogCallbackHandle_t userHandle,
                    hipdnnSeverity_t /*severity*/,
                    const char* message)
{
    auto* capture = static_cast<LogCapture*>(userHandle);
    const std::lock_guard<std::mutex> lock(capture->mutex);
    if(!capture->active || capture->lines.size() >= MAX_LOG_LINES_PER_CASE || message == nullptr)
    {
        return;
    }
    std::string line(message);
    if(line.size() > MAX_LOG_LINE_LENGTH)
    {
        line.resize(MAX_LOG_LINE_LENGTH);
    }
    capture->lines.push_back(line);
}

hipdnnSeverity_t severityOf(const std::string& level)
{
    if(level == "info")
    {
        return HIPDNN_SEV_INFO;
    }
    if(level == "warn")
    {
        return HIPDNN_SEV_WARN;
    }
    return HIPDNN_SEV_ERROR;
}

double microsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

// ------------------------------------------------------------------ one case

struct CaseResult
{
    std::string graphOutcome;
    std::string outcome;
    std::string stage;
    std::string message;
    std::string realised;
    double tRealiseUs = 0.0;
    double tValidateUs = 0.0;
    double tLowerUs = 0.0;
    double tQueryUs = 0.0;
    double tTotalUs = 0.0;
    bool queryRan = false;
    std::string planBuild = "not-run";
    std::string planBuildMessage;
    double tPlanBuildUs = 0.0;
    std::vector<std::string> otherEngines;
    std::string otherEnginesStatus;
};

CaseResult runCase(const CaseInput& input,
                   const Options& opts,
                   const sdpa_case_flow::CaseContext& context,
                   hipdnnHandle_t handle,
                   int64_t engineId,
                   LogCapture* capture)
{
    CaseResult result;
    const auto caseStart = Clock::now();
    result.stage = "realise";

    try
    {
        auto stageStart = Clock::now();
        sdpa_case_flow::BuiltCase built;
        try
        {
            built = sdpa_case_flow::buildCaseGraph(input.row, input.overrides, context);
        }
        catch(const sdpa_case_flow::CaseRealisationError& e)
        {
            result.tRealiseUs = microsSince(stageStart);
            result.graphOutcome = OUTCOME_ERROR;
            result.message = e.what();
            result.tTotalUs = microsSince(caseStart);
            return result;
        }
        result.tRealiseUs = microsSince(stageStart);
        result.realised = built.realised;
        auto& graph = *built.graph;

        result.stage = "validate";
        stageStart = Clock::now();
        const auto validated = graph.validate();
        result.tValidateUs = microsSince(stageStart);
        if(!validated.is_good())
        {
            result.graphOutcome = OUTCOME_FRONTEND_REJECTED;
            result.message = validated.get_message();
            result.tTotalUs = microsSince(caseStart);
            return result;
        }

        result.stage = "lower";
        stageStart = Clock::now();
        const auto lowered = graph.build_operation_graph(handle);
        result.tLowerUs = microsSince(stageStart);
        if(!lowered.is_good())
        {
            result.graphOutcome = OUTCOME_FRONTEND_REJECTED;
            result.message = lowered.get_message();
            result.tTotalUs = microsSince(caseStart);
            return result;
        }

        result.stage = "query";
        if(capture != nullptr)
        {
            const std::lock_guard<std::mutex> lock(capture->mutex);
            capture->lines.clear();
            capture->active = true;
        }
        stageStart = Clock::now();
        const auto queried = graph.create_execution_plan_ext(engineId, {});
        result.tQueryUs = microsSince(stageStart);
        result.queryRan = true;
        if(capture != nullptr)
        {
            const std::lock_guard<std::mutex> lock(capture->mutex);
            capture->active = false;
        }

        if(queried.is_good())
        {
            result.graphOutcome = OUTCOME_ACCEPTED;
        }
        else
        {
            result.message = queried.get_message();
            result.graphOutcome = result.message.find(ENGINE_REFUSED_MARKER) != std::string::npos
                                      ? OUTCOME_NOT_APPLICABLE
                                      : OUTCOME_ERROR;
        }
        result.tTotalUs = microsSince(caseStart);

        if(opts.smokeBuild && result.graphOutcome == OUTCOME_ACCEPTED)
        {
            result.stage = "plan-build";
            stageStart = Clock::now();
            auto planResult = graph.check_support();
            if(planResult.is_good())
            {
                planResult = graph.build_plans();
            }
            result.tPlanBuildUs = microsSince(stageStart);
            result.planBuild = planResult.is_good() ? "ok" : "failed";
            result.planBuildMessage = planResult.get_message();
            result.stage = "query";
        }

        if(opts.otherEngines && result.graphOutcome == OUTCOME_NOT_APPLICABLE)
        {
            std::vector<int64_t> ranked;
            const auto rankedStatus = graph.get_ranked_engine_ids(ranked);
            result.otherEnginesStatus
                = rankedStatus.is_good() ? "ok" : "error: " + rankedStatus.get_message();
            for(const auto id : ranked)
            {
                result.otherEngines.push_back(engineName(handle, id));
            }
        }
    }
    catch(const std::exception& e)
    {
        if(capture != nullptr)
        {
            const std::lock_guard<std::mutex> lock(capture->mutex);
            capture->active = false;
        }
        result.graphOutcome = OUTCOME_ERROR;
        result.message = std::string("exception: ") + e.what();
        result.tTotalUs = microsSince(caseStart);
    }
    return result;
}

// ------------------------------------------------------------------ summary statistics

struct Stats
{
    std::vector<double> values;

    std::string json() const
    {
        JsonLine line;
        line.integer("count", static_cast<int64_t>(values.size()));
        if(values.empty())
        {
            return line.done();
        }
        auto sorted = values;
        std::sort(sorted.begin(), sorted.end());
        const auto rank = [&](double fraction) {
            const auto index
                = static_cast<size_t>(std::ceil(fraction * static_cast<double>(sorted.size())));
            return sorted[std::min(sorted.size() - 1, index == 0 ? 0 : index - 1)];
        };
        double sum = 0.0;
        for(const double v : sorted)
        {
            sum += v;
        }
        line.num("median_us", rank(0.5))
            .num("p95_us", rank(0.95))
            .num("min_us", sorted.front())
            .num("max_us", sorted.back())
            .num("mean_us", sum / static_cast<double>(sorted.size()));
        return line.done();
    }
};

} // namespace

int main(int argc, char** argv)
{
    const auto runStart = Clock::now();
    const Options opts = parseArgs(argc, argv);
    InputFile input = readInput(opts.input);
    if(opts.limit > 0 && input.cases.size() > static_cast<size_t>(opts.limit))
    {
        input.cases.resize(static_cast<size_t>(opts.limit));
    }
    const ClassValuesFile classValues = readClassValues(opts.classValues);

    std::ofstream out(opts.output, std::ios::trunc);
    if(!out)
    {
        inputError("cannot open output '" + opts.output + "'");
    }

    const DeviceInfo device = queryDevice();
    const sdpa_case_flow::CaseContext caseContext{
        classValues.values, opts.descaleForm, baseArch(device.arch)};

    std::vector<std::string> argvCopy(argv, argv + argc);
    JsonLine runRecord;
    runRecord.str("record", "run")
        .raw("argv", jsonStringArray(argvCopy))
        .str("hostname", hostName())
        .str("arch", device.arch)
        .str("device_name", device.name)
        .integer("device_count", device.deviceCount)
        .integer("device", device.device)
        .str("hip_error", device.hipError)
        .str("HIP_VISIBLE_DEVICES", envOr("HIP_VISIBLE_DEVICES", "<unset>"))
        .str("ROCR_VISIBLE_DEVICES", envOr("ROCR_VISIBLE_DEVICES", "<unset>"))
        .str("CUDA_VISIBLE_DEVICES", envOr("CUDA_VISIBLE_DEVICES", "<unset>"))
        .str("GPU_DEVICE_ORDINAL", envOr("GPU_DEVICE_ORDINAL", "<unset>"))
        .str("input", std::filesystem::absolute(opts.input).string())
        .str("input_sha256", input.sha256)
        .integer("input_rows", static_cast<int64_t>(input.dataRows))
        .integer("rows_used", static_cast<int64_t>(input.cases.size()))
        .str("class_values", std::filesystem::absolute(opts.classValues).string())
        .str("class_values_sha256", classValues.sha256)
        .raw("ignored_columns", jsonStringArray(input.ignoredColumns))
        .integer("repeat", opts.repeat)
        .str("descale_form",
             opts.descaleForm == sdpa_case_flow::DescaleForm::BY_VALUE ? "by-value" : "device")
        .str("engine_requested", opts.engine);
    for(const auto& [key, value] : opts.meta)
    {
        runRecord.str("meta_" + key, value);
    }

    int exitCode = 0;
    std::string fatalStage;
    std::string fatalMessage;

    if(!opts.expectArch.empty() && baseArch(device.arch) != opts.expectArch)
    {
        fatalStage = "arch";
        fatalMessage
            = "HIP reports arch '" + device.arch + "'; --expect-arch is '" + opts.expectArch + "'";
        exitCode = EXIT_ARCH_MISMATCH;
    }

    if(exitCode == 0 && !opts.pluginPaths.empty())
    {
        const auto status = hipdnn_frontend::setEnginePluginPaths(
            opts.pluginPaths, hipdnn_frontend::PluginLoadingMode::MODE_ABSOLUTE);
        if(!status.is_good())
        {
            fatalStage = "plugin-paths";
            fatalMessage = status.get_message();
        }
    }
    runRecord.raw("plugin_paths_requested", jsonStringArray(opts.pluginPaths));

    hipdnnHandle_t handle = nullptr;
    int64_t engineId = 0;
    if(exitCode == 0 && fatalStage.empty())
    {
        const auto handleStart = Clock::now();
        const auto status = hipdnnCreate(&handle);
        runRecord.num("first_handle_create_us", microsSince(handleStart));
        if(status != HIPDNN_STATUS_SUCCESS)
        {
            handle = nullptr;
            fatalStage = "handle";
            fatalMessage
                = "hipdnnCreate failed with status " + std::to_string(static_cast<int>(status));
        }
    }

    if(handle != nullptr)
    {
        std::vector<std::filesystem::path> loaded;
        const auto loadedStatus = hipdnn_frontend::getLoadedEnginePluginPaths(handle, loaded);
        std::vector<std::string> loadedText;
        for(const auto& path : loaded)
        {
            loadedText.push_back(path.string());
        }
        runRecord.raw("plugins_loaded", jsonStringArray(loadedText))
            .str("plugins_loaded_status",
                 loadedStatus.is_good() ? "ok" : loadedStatus.get_message());

        const auto byNameStatus
            = hipdnnGetEngineIdByName_ext(handle, opts.engine.c_str(), &engineId);
        const int64_t hashedId = hipdnn_data_sdk::utilities::engineNameToId(opts.engine);
        runRecord.str("engine_id_hashed", hipdnn_data_sdk::utilities::formatEngineIdHex(hashedId));
        if(byNameStatus != HIPDNN_STATUS_SUCCESS)
        {
            fatalStage = "engine";
            fatalMessage = "no loaded engine carries the name '" + opts.engine + "'";
            exitCode = EXIT_ENGINE_NOT_LOADED;
        }
        else
        {
            runRecord.str("engine_id", hipdnn_data_sdk::utilities::formatEngineIdHex(engineId))
                .str("engine_name_loaded", engineName(handle, engineId));
        }
    }

    LogCapture capture;
    LogCapture* capturePtr = nullptr;
    if(handle != nullptr && fatalStage.empty() && !opts.captureLog.empty())
    {
        const auto level = severityOf(opts.captureLog);
        const auto globalStatus = hipdnnBackendSetGlobalLogLevel_ext(level);
        const auto callbackStatus = hipdnnSetUserLogCallback_ext(
            captureLogLine, level, HIPDNN_LOG_CALLBACK_SYNC, &capture);
        runRecord.str("capture_log",
                      "level=" + opts.captureLog
                          + " global_status=" + std::to_string(static_cast<int>(globalStatus))
                          + " callback_status=" + std::to_string(static_cast<int>(callbackStatus)));
        if(callbackStatus == HIPDNN_STATUS_SUCCESS)
        {
            capturePtr = &capture;
        }
    }

    runRecord.str("fatal_stage", fatalStage).str("fatal_message", fatalMessage);
    out << runRecord.done() << "\n" << std::flush;

    std::map<std::string, int64_t> outcomeCounts;
    std::map<std::string, int64_t> graphOutcomeCounts;
    Stats queryAll;
    Stats queryAccepted;
    Stats queryRefused;
    Stats totalAll;
    double firstQueryUs = -1.0;
    double firstTotalUs = -1.0;
    int64_t probedRows = 0;
    int64_t acceptedRows = 0;
    int64_t seq = 0;

    for(size_t index = 0; index < input.cases.size(); ++index)
    {
        const auto& caseInput = input.cases[index];

        bool decidedAtExecute = caseInput.needsExecution;
        for(const auto& axis : caseInput.decidingAxes)
        {
            decidedAtExecute = decidedAtExecute || sdpa_case_flow::isExecuteLevelAxis(axis);
        }
        std::vector<std::string> unobserved;
        for(const auto& axis : sdpa_case_flow::modelAxisIds())
        {
            if(!sdpa_case_flow::isExecuteLevelAxis(axis))
            {
                continue;
            }
            const auto& value = caseInput.row.at(axis);
            if(!value.empty() && value != "-" && value != sdpa_case_flow::sharedBaseValue(axis))
            {
                unobserved.push_back(axis + "=" + value);
            }
        }

        for(int64_t rep = 0; rep < opts.repeat; ++rep, ++seq)
        {
            CaseResult result;
            if(!fatalStage.empty())
            {
                result.graphOutcome = OUTCOME_ERROR;
                result.stage = fatalStage;
                result.message = fatalMessage;
            }
            else
            {
                result = runCase(caseInput, opts, caseContext, handle, engineId, capturePtr);
            }
            result.outcome = decidedAtExecute ? OUTCOME_NOT_PROBED : result.graphOutcome;

            const bool firstQuery = seq == 0;
            if(firstQuery)
            {
                firstQueryUs = result.queryRan ? result.tQueryUs : -1.0;
                firstTotalUs = result.tTotalUs;
            }
            else
            {
                totalAll.values.push_back(result.tTotalUs);
                if(result.queryRan)
                {
                    queryAll.values.push_back(result.tQueryUs);
                    if(result.graphOutcome == OUTCOME_ACCEPTED)
                    {
                        queryAccepted.values.push_back(result.tQueryUs);
                    }
                    else if(result.graphOutcome == OUTCOME_NOT_APPLICABLE)
                    {
                        queryRefused.values.push_back(result.tQueryUs);
                    }
                }
            }

            ++outcomeCounts[result.outcome];
            ++graphOutcomeCounts[result.graphOutcome];
            if(result.outcome != OUTCOME_NOT_PROBED)
            {
                ++probedRows;
                if(result.outcome == OUTCOME_ACCEPTED)
                {
                    ++acceptedRows;
                }
            }

            std::vector<std::string> logLines;
            if(capturePtr != nullptr)
            {
                const std::lock_guard<std::mutex> lock(capturePtr->mutex);
                logLines = capturePtr->lines;
                capturePtr->lines.clear();
            }

            JsonLine line;
            line.str("record", "case")
                .integer("seq", seq)
                .str("case_id", caseInput.caseId)
                .integer("repeat", rep)
                .boolean("first_query", firstQuery)
                .str("arch", device.arch)
                .str("outcome", result.outcome)
                .str("graph_outcome", result.graphOutcome)
                .str("stage", result.stage)
                .str("message", result.message)
                .raw("unobserved_axes", jsonStringArray(unobserved))
                .str("realised", result.realised)
                .raw("overrides", jsonStringArray(caseInput.overrides))
                .num("t_realise_us", result.tRealiseUs)
                .num("t_validate_us", result.tValidateUs)
                .num("t_lower_us", result.tLowerUs)
                .num("t_query_us", result.tQueryUs)
                .num("t_total_us", result.tTotalUs)
                .str("plan_build", result.planBuild)
                .str("plan_build_message", result.planBuildMessage)
                .num("t_plan_build_us", result.tPlanBuildUs);
            if(opts.otherEngines)
            {
                line.raw("other_engines", jsonStringArray(result.otherEngines))
                    .str("other_engines_status", result.otherEnginesStatus);
            }
            if(capturePtr != nullptr)
            {
                line.raw("log_lines", jsonStringArray(logLines));
            }
            out << line.done() << "\n" << std::flush;
        }

        if((index + 1) % 1000 == 0)
        {
            std::cerr << "progress: " << (index + 1) << " / " << input.cases.size() << " cases\n";
        }
    }

    const bool noneAccepted = probedRows > 0 && acceptedRows == 0;
    if(exitCode == 0 && noneAccepted && !opts.allowNoAccepted)
    {
        exitCode = EXIT_NO_CASE_ACCEPTED;
    }

    std::string outcomeJson = "{";
    for(const auto& [name, count] : outcomeCounts)
    {
        outcomeJson += (outcomeJson.size() == 1 ? "" : ",") + jsonString(name) + ":"
                       + std::to_string(count);
    }
    outcomeJson += "}";
    std::string graphOutcomeJson = "{";
    for(const auto& [name, count] : graphOutcomeCounts)
    {
        graphOutcomeJson += (graphOutcomeJson.size() == 1 ? "" : ",") + jsonString(name) + ":"
                            + std::to_string(count);
    }
    graphOutcomeJson += "}";

    JsonLine summary;
    summary.str("record", "summary")
        .str("arch", device.arch)
        .integer("lines_written", seq)
        .integer("probed_lines", probedRows)
        .integer("accepted_lines", acceptedRows)
        .boolean("no_case_accepted", noneAccepted)
        .raw("outcomes", outcomeJson)
        .raw("graph_outcomes", graphOutcomeJson)
        .num("first_query_us", firstQueryUs)
        .num("first_total_us", firstTotalUs)
        .raw("query_us_excluding_first", queryAll.json())
        .raw("query_us_accepted_excluding_first", queryAccepted.json())
        .raw("query_us_not_applicable_excluding_first", queryRefused.json())
        .raw("total_us_excluding_first", totalAll.json())
        .num("wall_s", microsSince(runStart) / 1.0e6)
        .integer("exit_code", exitCode);
    out << summary.done() << "\n" << std::flush;

    if(capturePtr != nullptr)
    {
        static_cast<void>(hipdnnSetUserLogCallback_ext(
            captureLogLine, HIPDNN_SEV_OFF, HIPDNN_LOG_CALLBACK_SYNC, &capture));
    }
    if(handle != nullptr)
    {
        static_cast<void>(hipdnnDestroy(handle));
    }

    std::cerr << "cases: " << input.cases.size() << ", lines: " << seq
              << ", accepted: " << acceptedRows << ", exit: " << exitCode << "\n";
    return exitCode;
}
