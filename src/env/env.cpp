#include <iostream>

#include <cstring>

#include "env/env.h"
#include "utils/logger.h"

namespace env {
  #define ENV_OPTGEMM(x) "OPTGEMM_" x;

  static char LOGLEVEL[]  = ENV_OPTGEMM("LOG");

  char* strupr(char* str) {
    char *s = str;
    while (*s) {
      *s = toupper((unsigned char) *s);
      s++;
    }
    return s;
  }

  /**
   * getLogLevel() - Get LogLevel value from environment value of LOGLEVEL
   */
  LogLevel getLogLevel() {
    char *val = getenv(LOGLEVEL);
    if (val            == nullptr) return LogLevel::Nothing;
    strupr(val);
    if (strcmp(val, "INFO")  == 0) return LogLevel::Info;
    if (strcmp(val, "DEBUG") == 0) return LogLevel::Debug;
    Logger(LogLevel::Info) << "Invalid " << LOGLEVEL << "=" << val << std::endl;
    return LogLevel::Nothing;
  }
}