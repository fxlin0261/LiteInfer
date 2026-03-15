#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include "base/unicode.h"

TEST(test_unicode, len_utf8_reports_expected_width_from_leading_byte) {
    EXPECT_EQ(unicode_len_utf8('$'), 1U);
    EXPECT_EQ(unicode_len_utf8(static_cast<char>(0xC2)), 2U);
    EXPECT_EQ(unicode_len_utf8(static_cast<char>(0xE4)), 3U);
    EXPECT_EQ(unicode_len_utf8(static_cast<char>(0xF0)), 4U);
}

TEST(test_unicode, codepoint_roundtrip_preserves_multibyte_values_and_offset) {
    size_t offset = 0;
    const std::string ascii = "A";
    EXPECT_EQ(unicode_cpt_from_utf8(ascii, offset), static_cast<uint32_t>('A'));
    EXPECT_EQ(offset, 1U);

    const std::string ni = "\xE4\xBD\xA0";
    offset = 0;
    EXPECT_EQ(unicode_cpt_from_utf8(ni, offset), 0x4F60U);
    EXPECT_EQ(offset, ni.size());
    EXPECT_EQ(unicode_cpt_to_utf8(0x4F60U), ni);

    const std::string smile = "\xF0\x9F\x99\x82";
    offset = 0;
    EXPECT_EQ(unicode_cpt_from_utf8(smile, offset), 0x1F642U);
    EXPECT_EQ(offset, smile.size());
    EXPECT_EQ(unicode_cpt_to_utf8(0x1F642U), smile);
}

TEST(test_unicode, byte_conversion_roundtrip_keeps_original_byte) {
    for (uint8_t byte : {uint8_t{0}, uint8_t{32}, uint8_t{65}, uint8_t{255}}) {
        const std::string utf8 = unicode_byte_to_utf8(byte);
        EXPECT_EQ(unicode_utf8_to_byte(utf8), byte);
    }
}

TEST(test_unicode, flags_and_case_conversion_match_basic_categories) {
    const auto upper = unicode_cpt_flags("A");
    EXPECT_TRUE(upper.is_letter);
    EXPECT_TRUE(upper.is_uppercase);
    EXPECT_FALSE(upper.is_lowercase);

    const auto lower = unicode_cpt_flags("a");
    EXPECT_TRUE(lower.is_letter);
    EXPECT_TRUE(lower.is_lowercase);
    EXPECT_FALSE(lower.is_uppercase);

    const auto digit = unicode_cpt_flags("7");
    EXPECT_TRUE(digit.is_number);

    const auto space = unicode_cpt_flags(" ");
    EXPECT_TRUE(space.is_whitespace);

    EXPECT_EQ(unicode_tolower(static_cast<uint32_t>('A')), static_cast<uint32_t>('a'));
    EXPECT_EQ(unicode_tolower(static_cast<uint32_t>('a')), static_cast<uint32_t>('a'));
}
