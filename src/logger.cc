#include <stdio.h>
#include <stdarg.h>
#include "define.h"


// Initialize the global logging level
LogLevel global_log_level = LOG_LEVEL_INFO;

void log_message(LogLevel level, const char *format, ...) {
    if (level < global_log_level) {
        return; // Skip logging messages below the current level
    }

    const char *level_strings[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    va_list args;

    fprintf(stderr, "[%s] ", level_strings[level]);

    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fprintf(stderr, "\n");
}