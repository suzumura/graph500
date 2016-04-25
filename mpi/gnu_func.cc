/*
 * gnu_func.c
 *
 *  Created on: Apr 6, 2012
 *      Author: koji
 */

#include <stdint.h>

int32_t my_sync_fetch_and_add(volatile int32_t* ptr, int32_t n) {
	return __sync_fetch_and_add(ptr, n);
}
int64_t my_sync_fetch_and_add(volatile int64_t* ptr, int64_t n) {
	return __sync_fetch_and_add(ptr, n);
}

int32_t my_sync_add_and_fetch(volatile int32_t* ptr, int32_t n) {
	return __sync_add_and_fetch(ptr, n);
}
int64_t my_sync_add_and_fetch(volatile int64_t* ptr, int64_t n) {
	return __sync_add_and_fetch(ptr, n);
}

uint32_t my_sync_fetch_and_or(volatile uint32_t* ptr, uint32_t n) {
	return __sync_fetch_and_or(ptr, n);
}
uint64_t my_sync_fetch_and_or(volatile uint64_t* ptr, uint64_t n) {
	return __sync_fetch_and_or(ptr, n);
}

bool my_sync_bool_compare_and_swap(volatile int64_t* ptr, int64_t old_value, int64_t new_value) {
	return __sync_bool_compare_and_swap(ptr, old_value, new_value);
}
bool my_sync_bool_compare_and_swap(volatile int32_t* ptr, int32_t old_value, int32_t new_value) {
	return __sync_bool_compare_and_swap(ptr, old_value, new_value);
}

