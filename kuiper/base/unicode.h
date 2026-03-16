#pragma once

#include <string>
#include <vector>
#include "base/unicode_byte_fallback.h"
#include "base/unicode_props.h"
#include "base/unicode_utf8.h"

std::vector<std::string> unicode_regex_split(const std::string& text,
                                             const std::vector<std::string>& regex_exprs);
