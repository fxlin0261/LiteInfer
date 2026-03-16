#include "base/unicode.h"
#include <cstdint>
#include <cstdio>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::wstring unicode_wstring_from_utf8(const std::string& s) {
    const auto cps = unicode_cpts_from_utf8(s);
    std::wstring result;
    result.reserve(cps.size());
    for (const auto cp : cps) {
        if (cp > 0x10FFFF || (0xD800 <= cp && cp <= 0xDFFF)) {
            throw std::invalid_argument("invalid codepoint");
        }

        if constexpr (sizeof(wchar_t) >= 4) {
            result.push_back(static_cast<wchar_t>(cp));
            continue;
        }

        if (cp <= 0xFFFF) {
            result.push_back(static_cast<wchar_t>(cp));
            continue;
        }

        const auto codepoint = cp - 0x10000;
        result.push_back(static_cast<wchar_t>(0xD800 + (codepoint >> 10)));
        result.push_back(static_cast<wchar_t>(0xDC00 + (codepoint & 0x3FF)));
    }
    return result;
}

std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string>& bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    bpe_encoded_words.reserve(bpe_words.size());
    for (const auto& word : bpe_words) {
        std::string encoded_token;
        for (const auto byte : word) {
            encoded_token += unicode_byte_to_utf8(static_cast<uint8_t>(byte));
        }
        bpe_encoded_words.emplace_back(std::move(encoded_token));
    }
    return bpe_encoded_words;
}

std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string& text,
                                                    const std::vector<size_t>& offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (const auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        const auto get_cpt = [&](const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        const auto get_flags = [&](const size_t pos) -> codepoint_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags(cpts[pos])
                                                           : codepoint_flags{};
        };

        size_t prev_end = offset_ini;
        const auto add_token = [&](const size_t end) -> size_t {
            const size_t len = end - prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end;) {
            const uint32_t cpt = get_cpt(pos);
            const auto flags = get_flags(pos);

            if (cpt == '\'' && pos + 1 < offset_end) {
                const uint32_t cpt_next = get_cpt(pos + 1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += add_token(pos + 2);
                    continue;
                }
                if (pos + 2 < offset_end) {
                    const uint32_t cpt_next_next = get_cpt(pos + 2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += add_token(pos + 3);
                        continue;
                    }
                }
            }

            auto flags2 = (cpt == ' ' ? get_flags(pos + 1) : flags);
            if (flags2.is_letter) {
                pos += (cpt == ' ');
                while (flags2.is_letter) {
                    flags2 = get_flags(++pos);
                }
                add_token(pos);
                continue;
            }
            if (flags2.is_number) {
                pos += (cpt == ' ');
                while (flags2.is_number) {
                    flags2 = get_flags(++pos);
                }
                add_token(pos);
                continue;
            }
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) &&
                       flags2.as_uint()) {
                    flags2 = get_flags(++pos);
                }
                add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            while (get_flags(pos + num_whitespaces).is_whitespace) {
                num_whitespaces++;
            }

            if (num_whitespaces > 1 && get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                add_token(pos);
                continue;
            }

            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                add_token(pos);
                continue;
            }

            add_token(++pos);
        }
    }

    return bpe_offsets;
}

std::vector<size_t> unicode_regex_split_custom_llama3(const std::string& text,
                                                      const std::vector<size_t>& offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    size_t start = 0;
    for (const auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        const auto get_cpt = [&](const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        const auto get_flags = [&](const size_t pos) -> codepoint_flags {
            return (offset_ini <= pos && pos < offset_end) ? unicode_cpt_flags(cpts[pos])
                                                           : codepoint_flags{};
        };

        size_t prev_end = offset_ini;
        const auto add_token = [&](const size_t end) -> size_t {
            const size_t len = end - prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end;) {
            const uint32_t cpt = get_cpt(pos);
            const auto flags = get_flags(pos);

            if (cpt == '\'' && pos + 1 < offset_end) {
                const uint32_t cpt_next = unicode_tolower(get_cpt(pos + 1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += add_token(pos + 2);
                    continue;
                }
                if (pos + 2 < offset_end) {
                    const uint32_t cpt_next_next = unicode_tolower(get_cpt(pos + 2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += add_token(pos + 3);
                        continue;
                    }
                }
            }

            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || get_flags(pos + 1).is_letter) {
                    pos++;
                    while (get_flags(pos).is_letter) {
                        pos++;
                    }
                    add_token(pos);
                    continue;
                }
            }

            if (flags.is_number) {
                size_t ini = pos;
                while (get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        add_token(pos);
                        ini = pos;
                    }
                }
                add_token(pos);
                continue;
            }

            auto flags2 = (cpt == ' ' ? get_flags(pos + 1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) &&
                       flags2.as_uint()) {
                    flags2 = get_flags(++pos);
                }
                uint32_t cpt2 = get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = get_cpt(++pos);
                }
                add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (get_flags(pos + num_whitespaces).is_whitespace) {
                const uint32_t cpt2 = get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                add_token(pos);
                continue;
            }

            if (num_whitespaces > 1 && get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                add_token(pos);
                continue;
            }

            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                add_token(pos);
                continue;
            }

            add_token(++pos);
        }
    }

    return bpe_offsets;
}

std::vector<size_t> unicode_regex_split_stl(const std::wstring& wtext,
                                            const std::wstring& regex_expr,
                                            const std::vector<size_t>& offsets) {
    std::wregex expr(regex_expr);
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    size_t start = 0;
    for (const auto offset : offsets) {
        std::wcregex_iterator it(wtext.data() + start, wtext.data() + start + offset, expr);
        std::wcregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            const std::wcmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < static_cast<int64_t>(offset)) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

std::vector<size_t> unicode_regex_split_stl(const std::string& text,
                                            const std::string& regex_expr,
                                            const std::vector<size_t>& offsets) {
    std::regex expr(regex_expr);
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());
    size_t start = 0;
    for (const auto offset : offsets) {
        std::cregex_iterator it(text.data() + start, text.data() + start + offset, expr);
        std::cregex_iterator end;

        int64_t start_idx = 0;
        while (it != end) {
            const std::cmatch match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < static_cast<int64_t>(offset)) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

std::vector<size_t> unicode_regex_split_custom(const std::string& text,
                                               const std::string& regex_expr,
                                               const std::vector<size_t>& offsets) {
    if (regex_expr ==
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        return unicode_regex_split_custom_gpt2(text, offsets);
    }

    if (regex_expr ==
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
            "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
        regex_expr ==
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]"
            "?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        return unicode_regex_split_custom_llama3(text, offsets);
    }

    return {};
}

}  // namespace

std::vector<std::string> unicode_regex_split(const std::string& text,
                                             const std::vector<std::string>& regex_exprs) {
    static const std::map<std::string, int> k_ucat_enum = {
        {"\\p{N}", codepoint_flags::NUMBER},
        {"\\p{L}", codepoint_flags::LETTER},
        {"\\p{P}", codepoint_flags::PUNCTUATION},
    };

    static const std::map<int, int> k_ucat_cpt = {
        {codepoint_flags::NUMBER, 0xD1},
        {codepoint_flags::LETTER, 0xD2},
        {codepoint_flags::PUNCTUATION, 0xD3},
    };

    static const std::map<int, std::string> k_ucat_map = {
        {codepoint_flags::NUMBER, "\x30-\x39"},
        {codepoint_flags::LETTER, "\x41-\x5A\x61-\x7A"},
        {codepoint_flags::PUNCTUATION,
         "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D"},
    };

    bool need_collapse = false;
    for (const auto& regex_expr : regex_exprs) {
        for (const auto& ucat : k_ucat_enum) {
            if (regex_expr.find(ucat.first) != std::string::npos) {
                need_collapse = true;
                break;
            }
        }
    }

    const auto cpts = unicode_cpts_from_utf8(text);

    std::string text_collapsed;
    if (need_collapse) {
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto flags = unicode_cpt_flags(cpts[i]);
            if (flags.is_whitespace) {
                text_collapsed[i] = static_cast<char>(0x0B);
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = static_cast<char>(0xD0);
            }
        }
    }

    std::vector<size_t> bpe_offsets = {cpts.size()};

    for (const auto& regex_expr : regex_exprs) {
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);
        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        try {
            bool use_collapsed = false;
            for (const auto& ucat : k_ucat_enum) {
                if (regex_expr.find(ucat.first) != std::string::npos) {
                    use_collapsed = true;
                    break;
                }
            }

            if (use_collapsed) {
                const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);
                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error(
                            "Regex includes both unicode categories and non-ASCII characters - not "
                            "supported");
                    }
                }

                std::string regex_expr_collapsed;
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    if (regex_expr[i + 0] == '\\' && i + 4 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' && regex_expr[i + 2] == '{' &&
                        regex_expr[i + 4] == '}') {
                        const auto pat = regex_expr.substr(i, 5);
                        if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                            if (!inside) {
                                regex_expr_collapsed += '[';
                            }
                            regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                            regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                            if (!inside) {
                                regex_expr_collapsed += ']';
                            }
                            i += 4;
                            continue;
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                bpe_offsets =
                    unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                const auto wregex_expr = unicode_wstring_from_utf8(regex_expr);
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && unicode_cpt_flags(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }

                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error& e) {
            std::fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            std::fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size());

    size_t start = 0;
    for (const auto offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}
