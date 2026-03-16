#include "base/unicode_props.h"
#include <algorithm>
#include <cassert>
#include <vector>
#include "base/unicode-data.h"
#include "base/unicode_utf8.h"

namespace {
std::vector<codepoint_flags> unicode_cpt_flags_array() {
    std::vector<codepoint_flags> cpt_flags(MAX_CODEPOINTS, codepoint_flags::UNDEFINED);
    assert(unicode_ranges_flags.front().first == 0);
    assert(unicode_ranges_flags.back().first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags[i - 1];
        const auto range_end = unicode_ranges_flags[i];
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (const auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (const auto& p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (const auto& p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (const auto& range : unicode_ranges_nfd) {
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}
}  // namespace

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t>& cpts) {
    const auto comp = [](const uint32_t cpt, const range_nfd& range) { return cpt < range.first; };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        const auto it =
            std::upper_bound(unicode_ranges_nfd.cbegin(), unicode_ranges_nfd.cend(), cpt, comp) -
            1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

codepoint_flags unicode_cpt_flags(uint32_t cp) {
    static const codepoint_flags undef(codepoint_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cp < cpt_flags.size() ? cpt_flags[cp] : undef;
}

codepoint_flags unicode_cpt_flags(const std::string& utf8) {
    static const codepoint_flags undef(codepoint_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;
    }
    size_t offset = 0;
    return unicode_cpt_flags(unicode_cpt_from_utf8(utf8, offset));
}

uint32_t unicode_tolower(uint32_t cp) {
    const auto it = unicode_map_lowercase.find(cp);
    return it == unicode_map_lowercase.end() ? cp : it->second;
}
