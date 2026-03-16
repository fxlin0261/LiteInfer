#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct codepoint_flags {
    enum {
        UNDEFINED = 0x0001,
        NUMBER = 0x0002,       // regex: \p{N}
        LETTER = 0x0004,       // regex: \p{L}
        SEPARATOR = 0x0008,    // regex: \p{Z}
        ACCENT_MARK = 0x0010,  // regex: \p{M}
        PUNCTUATION = 0x0020,  // regex: \p{P}
        SYMBOL = 0x0040,       // regex: \p{S}
        CONTROL = 0x0080,      // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
    };
    uint16_t is_undefined : 1;
    uint16_t is_number : 1;
    uint16_t is_letter : 1;
    uint16_t is_separator : 1;
    uint16_t is_accent_mark : 1;
    uint16_t is_punctuation : 1;
    uint16_t is_symbol : 1;
    uint16_t is_control : 1;
    uint16_t is_whitespace : 1;
    uint16_t is_lowercase : 1;
    uint16_t is_uppercase : 1;
    uint16_t is_nfd : 1;
    inline codepoint_flags(const uint16_t flags = 0) { *reinterpret_cast<uint16_t*>(this) = flags; }
    inline uint16_t as_uint() const { return *reinterpret_cast<const uint16_t*>(this); }
    inline uint16_t category_flag() const { return this->as_uint() & MASK_CATEGORIES; }
};
std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t>& cpts);
codepoint_flags unicode_cpt_flags(uint32_t cp);
codepoint_flags unicode_cpt_flags(const std::string& utf8);
uint32_t unicode_tolower(uint32_t cp);
