#pragma once

enum LogLevel {
  Nothing = 0,
  Info = 1,
  Debug = 2
};

namespace env {
  LogLevel getLogLevel();
}