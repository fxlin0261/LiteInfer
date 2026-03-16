#include "base/unicode_utf8.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    const uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

uint32_t unicode_cpt_from_utf8(const std::string& utf8, size_t& offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        const auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        const auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) ||
            !((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        const auto result = ((utf8[offset + 0] & 0x0f) << 12) |
                            ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) ||
            !((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        const auto result = ((utf8[offset + 0] & 0x07) << 18) |
                            ((utf8[offset + 1] & 0x3f) << 12) |
                            ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("failed to convert utf8 to codepoint");
}

std::string unicode_cpt_to_utf8(uint32_t cp) {
    std::string result;

    if (cp <= 0x7f) {
        result.push_back(cp);
        return result;
    }
    if (0x80 <= cp && cp <= 0x7ff) {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x800 <= cp && cp <= 0xffff) {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }
    if (0x10000 <= cp && cp <= 0x10ffff) {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& utf8) {
    std::vector<uint32_t> result;
    result.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        result.push_back(unicode_cpt_from_utf8(utf8, offset));
    }
    return result;
}
