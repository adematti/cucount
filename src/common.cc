#include <stdlib.h>
#include "common.h"

// Wrapper for calloc with error handling
void* my_calloc(size_t num, size_t size) {
    void* ptr = calloc(num, size);
    if (!ptr) {
        log_message(LOG_LEVEL_ERROR, "Memory allocation failed in my_calloc for %zu elements of size %zu.", num, size);
        exit(EXIT_FAILURE); // Exit the program on allocation failure
    }
    return ptr;
}

// Wrapper for malloc with error handling
void* my_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        log_message(LOG_LEVEL_ERROR, "Memory allocation failed in my_malloc for size %zu.", size);
        exit(EXIT_FAILURE); // Exit the program on allocation failure
    }
    return ptr;
}
