#pragma once

#include <cstdint>
#include <string>

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string& utf8);
