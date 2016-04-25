/*
 * utils_fwd.h
 *
 *  Created on: Dec 15, 2011
 *      Author: koji
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <stdint.h>
#include <assert.h>


// for sorting
#include <algorithm>
#include <functional>

using std::ptrdiff_t;

#ifdef __cplusplus
#define restrict __restrict__
#endif

//-------------------------------------------------------------//
// For generic typing
//-------------------------------------------------------------//

template <typename T> struct MpiTypeOf { };

//-------------------------------------------------------------//
// Bit manipulation functions
//-------------------------------------------------------------//

#if defined(__INTEL_COMPILER)

#define get_msb_index _bit_scan_reverse
#define get_lsb_index _bit_scan

#elif defined(__GNUC__)
#define NLEADING_ZERO_BITS __builtin_clz
#define NLEADING_ZERO_BITSL __builtin_clzl
#define NLEADING_ZERO_BITSLL __builtin_clzll

// If value = 0, the result is undefined.
inline int get_msb_index(int64_t value) {
	assert (value != 0);
	return (sizeof(value)*8-1) - INT64_C(NLEADING_ZERO_BITS)(value);
}

#undef NLEADING_ZERO_BITS
#undef NLEADING_ZERO_BITSL
#undef NLEADING_ZERO_BITSLL
#endif // #ifdef __GNUC__

#ifdef __sparc_v9__

inline int __builtin_popcountl_asm(uint64_t n) {
	int c;
	__asm__(
			"popc %1, %0\n\t"
			:"=r"(c)
			:"r"(n)
			);
	assert(__builtin_popcountl(n) == c);
	return c;
}

inline int __builtin_popcount_asm(uint32_t n) {
	int c;
	__asm__(
			"popc %1, %0\n\t"
			:"=r"(c)
			:"r"(n)
			);
	assert(__builtin_popcount(n) == c);
	return c;
}

inline int __builtin_ctzl_asm(uint64_t n) {
	return __builtin_popcountl_asm((n&(-n))-1);
}

inline int __builtin_ctz_asm(uint32_t n) {
	return __builtin_popcount_asm((n&(-n))-1);
}

#define __builtin_popcountl __builtin_popcountl_asm
#define __builtin_popcount32bit __builtin_popcount_asm
#define __builtin_popcount64bit __builtin_popcountl_asm

#define __builtin_ctzl __builtin_ctzl_asm
#define __builtin_ctz32bit __builtin_ctz_asm
#define __builtin_ctz64bit __builtin_ctzl_asm

#if ENABLE_INLINE_ATOMICS
inline int32_t __sync_fetch_and_add(volatile int32_t* ptr, int32_t n) {
	int32_t old_value;
	__asm__ (
		"ld     [%2], %0\n"
	"1:\n\t"
		"add    %0, %3, %%l0\n\t"
			"membar 15\n\t"
		"cas    [%2], %0, %%l0\n\t"
		"cmp    %0, %%l0\n\t"
		"bne,a,pn %%icc, 1b\n\t"
		"mov    %%l0, %0\n\t"
		:"=&r"(old_value),"=m"(*ptr)
		:"r"(ptr),"r"(n)
		:"%l0","cc"
	);
	return old_value;
}
inline int64_t __sync_fetch_and_add(volatile int64_t* ptr, int64_t n) {
	int64_t old_value;
	__asm__ (
		"ldx    [%2], %0\n"
	"1:\n\t"
		"add    %0, %3, %%l0\n\t"
			"membar 15\n\t"
		"casx  [%2], %0, %%l0\n\t"
		"cmp    %0, %%l0\n\t"
		"bne,a,pn %%xcc, 1b\n\t"
		"mov    %%l0, %0\n\t"
		:"=&r"(old_value),"=m"(*ptr)
		:"r"(ptr),"r"(n)
		:"%l0","cc"
	);
	return old_value;
}

inline int32_t __sync_add_and_fetch(volatile int32_t* ptr, int32_t n) {
	int32_t old_value;
	__asm__ (
		"ld     [%2], %0\n"
	"1:\n\t"
		"add    %0, %3, %%l0\n\t"
			"membar 15\n\t"
		"cas    [%2], %0, %%l0\n\t"
		"cmp    %0, %%l0\n\t"
		"bne,a,pn %%icc, 1b\n\t"
		"mov    %%l0, %0\n\t"
		:"=&r"(old_value),"=m"(*ptr)
		:"r"(ptr),"r"(n)
		:"%l0","cc"
	);
	return old_value + n;
}
inline int64_t __sync_add_and_fetch(volatile int64_t* ptr, int64_t n) {
	int64_t old_value;
	__asm__ (
		"ldx    [%2], %0\n"
	"1:\n\t"
		"add    %0, %3, %%l0\n\t"
			"membar 15\n\t"
		"casx  [%2], %0, %%l0\n\t"
		"cmp    %0, %%l0\n\t"
		"bne,a,pn %%xcc, 1b\n\t"
		"mov    %%l0, %0\n\t"
		:"=&r"(old_value),"=m"(*ptr)
		:"r"(ptr),"r"(n)
		:"%l0","cc"
	);
	return old_value + n;
}

inline uint32_t __sync_fetch_and_or(volatile uint32_t* ptr, uint32_t n) {
	int32_t old_value;
	__asm__ (
		"ld     [%2], %0\n"
	"1:\n\t"
		"or     %0, %3, %%l0\n\t"
			"membar 15\n\t"
		"cas    [%2], %0, %%l0\n\t"
		"cmp    %0, %%l0\n\t"
		"bne,a,pn %%icc, 1b\n\t"
		"mov    %%l0, %0\n\t"
		:"=&r"(old_value),"=m"(*ptr)
		:"r"(ptr),"r"(n)
		:"%l0","cc"
	);
	return old_value;
}
inline uint64_t __sync_fetch_and_or(volatile uint64_t* ptr, uint64_t n) {
	int64_t old_value;
	__asm__ (
		"ldx    [%2], %0\n"
	"1:\n\t"
		"or     %0, %3, %%l0\n\t"
			"membar 15\n\t"
		"casx  [%2], %0, %%l0\n\t"
		"cmp    %0, %%l0\n\t"
		"bne,a,pn %%xcc, 1b\n\t"
		"mov    %%l0, %0\n\t"
		:"=&r"(old_value),"=m"(*ptr)
		:"r"(ptr),"r"(n)
		:"%l0","cc"
	);
	return old_value;
}

inline bool __sync_bool_compare_and_swap(volatile int32_t* ptr, int32_t old_value, int32_t new_value) {
	bool ret;
	__asm__ (
		"membar 15\n\t"
		"cas    [%2], %3, %4\n\t"
		"cmp    %3, %4\n\t"
		"or     %%g0, 1, %0\n\t"
		"movne  %%icc, 0, %0\n\t"
		:"=r"(ret),"=m"(*ptr)
		:"r"(ptr),"r"(old_value),"r"(new_value)
		:"cc"
	);
	return ret;
}
inline bool __sync_bool_compare_and_swap(volatile int64_t* ptr, int64_t old_value, int64_t new_value) {
	bool ret;
	__asm__ (
		"membar 15\n\t"
		"casx  [%2], %3, %4\n\t"
		"cmp    %3, %4\n\t"
		"or     %%g0, 1, %0\n\t"
		"movne  %%xcc, 0, %0\n\t"
		:"=r"(ret),"=m"(*ptr)
		:"r"(ptr),"r"(old_value),"r"(new_value)
		:"cc"
	);
	return ret;
}
#endif

// Since Fujitsu compiler does not support __sync_synchronize.
// We define here.
static __thread uint this_is_not_used_ = 0;
#define __sync_synchronize() do { __sync_fetch_and_or(&this_is_not_used_, 1); } while(false)

#define NEXT_BIT(flags__, flag__, mask__, idx__) do {\
	flag__ = flags__ & (-flags__);\
	mask__ = flag__ - 1;\
	flags__ &= ~flag__;\
	idx__ = __builtin_popcountl(mask__); } while(false)\

#else // #ifdef __sparc_v9__

#define NEXT_BIT(flags__, flag__, mask__, idx__) do {\
	idx__ = __builtin_ctzl(flags__);\
	flag__ = BitmapType(1) << idx__;\
	mask__ = flag__ - 1;\
	flags__ &= ~flag__; } while(false)\

#define __builtin_popcount32bit __builtin_popcount
#define __builtin_popcount64bit __builtin_popcountl

#define __builtin_ctz32bit __builtin_ctz
#define __builtin_ctz64bit __builtin_ctzl

#endif // #ifdef __sparc_v9__

// Clear the bit size of each built-in function.
#define __builtin_popcount THIS_IS_FOR_32BIT_INT_AND_NOT_64BIT
#define __builtin_ctz THIS_IS_FOR_32BIT_INT_AND_NOT_64BIT

//-------------------------------------------------------------//
// Memory Allocation
//-------------------------------------------------------------//

void* xMPI_Alloc_mem(size_t nbytes);
void* cache_aligned_xcalloc(const size_t size);
void* cache_aligned_xmalloc(const size_t size);
void* page_aligned_xcalloc(const size_t size);
void* page_aligned_xmalloc(const size_t size);

//-------------------------------------------------------------//
// Sort
//-------------------------------------------------------------//

template <typename T1, typename T2>
class pointer_pair_value
{
public:
	T1 v1;
	T2 v2;
	pointer_pair_value(T1& v1_, T2& v2_)
		: v1(v1_)
		, v2(v2_)
	{ }
	operator T1 () const { return v1; }
	pointer_pair_value& operator=(const pointer_pair_value& o) { v1 = o.v1; v2 = o.v2; return *this; }
};

template <typename T1, typename T2>
class pointer_pair_reference
{
public:
	T1& v1;
	T2& v2;
	pointer_pair_reference(T1& v1_, T2& v2_)
		: v1(v1_)
		, v2(v2_)
	{ }
	operator T1 () const { return v1; }
	operator pointer_pair_value<T1, T2> () const { return pointer_pair_value<T1, T2>(v1, v2); }
	pointer_pair_reference& operator=(const pointer_pair_reference& o) { v1 = o.v1; v2 = o.v2; return *this; }
	pointer_pair_reference& operator=(const pointer_pair_value<T1, T2>& o) { v1 = o.v1; v2 = o.v2; return *this; }
};

template <typename T1, typename T2>
void swap(pointer_pair_reference<T1, T2> r1, pointer_pair_reference<T1, T2> r2)
{
	pointer_pair_value<T1, T2> tmp = r1;
	r1 = r2;
	r2 = tmp;
}

template <typename T1, typename T2>
class pointer_pair_iterator
	: public std::iterator<std::random_access_iterator_tag,
		pointer_pair_value<T1, T2>,
		ptrdiff_t,
		pointer_pair_value<T1, T2>*,
		pointer_pair_reference<T1, T2> >
{
public:
	T1* first;
	T2* second;

	typedef ptrdiff_t difference_type;
	typedef pointer_pair_reference<T1, T2> reference;

	pointer_pair_iterator(T1* v1, T2* v2) : first(v1) , second(v2) { }

	// Can be default-constructed
	pointer_pair_iterator() { }
	// Accepts equality/inequality comparisons
	bool operator==(const pointer_pair_iterator& ot) { return first == ot.first; }
	bool operator!=(const pointer_pair_iterator& ot) { return first != ot.first; }
	// Can be dereferenced
	reference operator*(){ return reference(*first, *second); }
	// Can be incremented and decremented
	pointer_pair_iterator operator++(int) { pointer_pair_iterator old(*this); ++first; ++second; return old; }
	pointer_pair_iterator operator--(int) { pointer_pair_iterator old(*this); --first; --second; return old; }
	pointer_pair_iterator& operator++() { ++first; ++second; return *this; }
	pointer_pair_iterator& operator--() { --first; --second; return *this; }
	// Supports arithmetic operators + and - between an iterator and an integer value, or subtracting an iterator from another
	pointer_pair_iterator operator+(const difference_type n) { pointer_pair_iterator t(first+n, second+n); return t; }
	pointer_pair_iterator operator-(const difference_type n) { pointer_pair_iterator t(first-n, second-n); return t; }
	size_t operator-(const pointer_pair_iterator& o) { return first - o.first; }
	// Supports inequality comparisons (<, >, <= and >=) between iterators
	bool operator<(const pointer_pair_iterator& o) { return first < o.first; }
	bool operator>(const pointer_pair_iterator& o) { return first > o.first; }
	bool operator<=(const pointer_pair_iterator& o) { return first <= o.first; }
	bool operator>=(const pointer_pair_iterator& o) { return first >= o.first; }
	// Supports compound assinment operations += and -=
	pointer_pair_iterator& operator+=(const difference_type n) { first += n; second += n; return *this; }
	pointer_pair_iterator& operator-=(const difference_type n) { first -= n; second -= n; return *this; }
	// Supports offset dereference operator ([])
	reference operator[](const difference_type n) { return reference(first[n], second[n]); }
};

template<typename IterKey, typename IterValue>
void sort2(IterKey* begin_key, IterValue* begin_value, size_t count)
{
	pointer_pair_iterator<IterKey, IterValue>
		begin(begin_key, begin_value), end(begin_key + count, begin_value + count);
	std::sort(begin, end);
}

template<typename IterKey, typename IterValue, typename Compare>
void sort2(IterKey* begin_key, IterValue* begin_value, size_t count, Compare comp)
{
	pointer_pair_iterator<IterKey, IterValue>
		begin(begin_key, begin_value), end(begin_key + count, begin_value + count);
	std::sort(begin, end, comp);
}

//-------------------------------------------------------------//
// Other functions
//-------------------------------------------------------------//

struct SeparatedId {
	uint64_t value;

	explicit SeparatedId(uint64_t v) : value(v) { }
	SeparatedId(int high, uint64_t low, int lgl)
		: value((uint64_t(high) << lgl) | low) { }
	uint64_t raw() const { return value; }
	uint64_t compact(int lgl, int64_t L) const { return high(lgl) * L + low(lgl); }
	int high(int lgl) const { return value >> lgl; }
	uint64_t low(int lgl) const { return value & ((uint64_t(1) << lgl) - 1); }
	int64_t swaplow(int mid, int lgl) {
		int64_t low_v = value >> (mid + lgl);
		int64_t mid_v = (value >> lgl) & ((1 << mid) - 1);
		return (mid_v << lgl) | low_v;
	}
};

struct MPI_INFO_ON_GPU {
	int rank;
	int size;
	int rank_2d;
	int rank_2dr;
	int rank_2dc;
};

int64_t get_time_in_microsecond();
FILE* get_imd_out_file();

#if ENABLE_FUJI_PROF
extern "C" {
void fapp_start(const char *, int , int);
void fapp_stop(const  char *, int , int);
void start_collection(const char *);
void stop_collection(const char *);
}
#endif

#endif /* UTILS_HPP_ */
