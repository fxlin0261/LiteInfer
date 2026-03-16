#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8(uint32_t cp);
uint32_t unicode_cpt_from_utf8(const std::string& utf8, size_t& offset);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& utf8);
