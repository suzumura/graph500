/*
 * utils.hpp
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 */

#ifndef UTILS_IMPL_HPP_
#define UTILS_IMPL_HPP_

#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>

// for affinity setting //
#include <unistd.h>

#include <sched.h>
#if NUMA_BIND
#include <numa.h>
#endif

#include <omp.h>
#if BACKTRACE_ON_SIGNAL
#include <signal.h>
#endif
#if ENABLE_FJMPI
#include <mpi-ext.h>
#endif

#include <sys/types.h>
#include <sys/time.h>
#include <sys/shm.h>

#include <algorithm>
#include <vector>
#include <deque>

#include "mpi_workarounds.h"
#include "utils_core.h"
#include "primitives.hpp"
#if CUDA_ENABLED
#include "gpu_host.hpp"
#endif

#if VTRACE
#include "vt_user.h"
#define USER_START(s) VT_USER_START(#s)
#define USER_END(s) VT_USER_END(#s)
#define TRACER(s) VT_TRACER(#s)
#define CTRACER(s)
#elif SCOREP
#include <scorep/SCOREP_User.h>
#define USER_START(s) SCOREP_USER_REGION_DEFINE( scorep_region_##s ); SCOREP_USER_REGION_BEGIN( scorep_region_##s, #s, SCOREP_USER_REGION_TYPE_COMMON )
#define USER_END(s) SCOREP_USER_REGION_END( scorep_region_##s )
#define TRACER(s) SCOREP_USER_REGION( #s, SCOREP_USER_REGION_TYPE_COMMON )
#define CTRACER(s)
#elif BACKTRACE_ON_SIGNAL
extern "C" void user_defined_proc(const int *FLAG, const char *NAME, const int *LINE, const int *THREAD);

struct ScopedRegion {
	const char* name_;
	int line_;
	ScopedRegion(const char* name, int line) {
		int flag = 102;
		name_ = name;
		line_ = line;
		user_defined_proc(&flag, name_, &line_, NULL);
	}
	~ScopedRegion() {
		int flag = 103;
		user_defined_proc(&flag, name_, &line_, NULL);
	}
};

#define USER_START(s) do { int line = __LINE__; int flag = 102;\
		user_defined_proc(&flag, __FILE__, &line, NULL); } while (false)
#define USER_END(s) do { int line = __LINE__; int flag = 103;\
		user_defined_proc(&flag, __FILE__, &line, NULL); } while (false)
#define TRACER(s) ScopedRegion my_trace_obj(__FILE__, __LINE__)

#else // #if VTRACE
#define USER_START(s)
#define USER_END(s)
#define TRACER(s)
#define CTRACER(s)
#endif // #if VTRACE


#if VERVOSE_MODE
#define VERVOSE(s) s
#else
#define VERVOSE(s)
#endif

#if PROFILING_MODE
#define PROF(s) s
#else
#define PROF(s)
#endif

void print_with_prefix(const char* format, ...);

#if DEBUG_PRINT
#define DEBUG_PRINT_SWITCH_0(...)
#define DEBUG_PRINT_SWITCH_1(...) print_with_prefix(__VA_ARGS__)
#define DEBUG_PRINT_SWITCH_2(...) do{if(mpi.isMaster())print_with_prefix(__VA_ARGS__);}while(0)
#define MAKE_DEBUG_PRINT_SWITCH(val) MAKE_DEBUG_PRINT_SWITCH_ASSIGN(val)
#define MAKE_DEBUG_PRINT_SWITCH_ASSIGN(val) DEBUG_PRINT_SWITCH_ ## val
#define debug_print(prefix, ...) MAKE_DEBUG_PRINT_SWITCH\
	(DEBUG_PRINT_ ## prefix)(#prefix " " __VA_ARGS__)
#else
#define debug_print(prefix, ...)
#endif

#if BACKTRACE_ON_SIGNAL
namespace backtrace {
void start_thread();
void thread_join();
}
#endif

struct COMM_2D {
	MPI_Comm comm;
	int rank, rank_x, rank_y;
	int size, size_x, size_y;
	int* rank_map; // Index: rank_x + rank_y * size_x
};

static void swap(COMM_2D& a, COMM_2D& b) {
	COMM_2D tmp = b;
	b = a;
	a = tmp;
}

struct MPI_GLOBALS {
	int rank;
	int size;
	int thread_level;

	// 2D
	int rank_2d;
	int rank_2dr;
	int rank_2dc;
	int size_2d; // = size
	int size_2dc; // = comm_2dr.size()
	int size_2dr; // = comm_2dc.size()
	MPI_Comm comm_2d;
	MPI_Comm comm_2dr; // = comm_x
	MPI_Comm comm_2dc;
	bool isRowMajor;

	// multi dimension
	COMM_2D comm_r;
	COMM_2D comm_c;
	bool isMultiDimAvailable;

	// for shared memory
	int rank_y;
	int rank_z;
	int size_y; // = comm_y.size()
	int size_z; // = comm_z.size()
	MPI_Comm comm_y;
	MPI_Comm comm_z;

	// utility method
	bool isMaster() const { return rank == 0; }
	bool isRmaster() const { return rank == size-1; }
	bool isYdimAvailable() const { return comm_y != comm_2dc; }
};

MPI_GLOBALS mpi;

//-------------------------------------------------------------//
// For generic typing
//-------------------------------------------------------------//

template <> struct MpiTypeOf<char> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<char>::type = MPI_CHAR;
template <> struct MpiTypeOf<short> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<short>::type = MPI_SHORT;
template <> struct MpiTypeOf<int> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<int>::type = MPI_INT;
template <> struct MpiTypeOf<long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<long>::type = MPI_LONG;
template <> struct MpiTypeOf<long long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<long long>::type = MPI_LONG_LONG;
template <> struct MpiTypeOf<float> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<float>::type = MPI_FLOAT;
template <> struct MpiTypeOf<double> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<double>::type = MPI_DOUBLE;
template <> struct MpiTypeOf<unsigned char> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned char>::type = MPI_UNSIGNED_CHAR;
template <> struct MpiTypeOf<unsigned short> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned short>::type = MPI_UNSIGNED_SHORT;
template <> struct MpiTypeOf<unsigned int> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned int>::type = MPI_UNSIGNED;
template <> struct MpiTypeOf<unsigned long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned long>::type = MPI_UNSIGNED_LONG;
template <> struct MpiTypeOf<unsigned long long> { static const MPI_Datatype type; };
const MPI_Datatype MpiTypeOf<unsigned long long>::type = MPI_UNSIGNED_LONG_LONG;


template <typename T> struct template_meta_helper { typedef void type; };

template <typename T> MPI_Datatype get_mpi_type(T& instance) {
	return MpiTypeOf<T>::type;
}

int64_t get_time_in_microsecond()
{
	struct timeval l;
	gettimeofday(&l, NULL);
	return ((int64_t)l.tv_sec*1000000 + l.tv_usec);
}

#if PRINT_WITH_TIME
struct GLOBAL_CLOCK {
	struct timeval l;
	int64_t clock_start;
	void init() {
		MPI_Barrier(MPI_COMM_WORLD);
		gettimeofday(&l, NULL);
		clock_start = ((int64_t)l.tv_sec*1000000 + l.tv_usec);
	}
	int64_t get() {
		return get_time_in_microsecond() - clock_start;
	}
};

GLOBAL_CLOCK global_clock;
#endif // #if PRINT_WITH_TIME

FILE* g_out_file = NULL;
FILE* get_imd_out_file() {
	if(mpi.size == 0) {
		// before MPI_Init()
		return stderr;
	}
	if(g_out_file == NULL) {
		char buf[100];
		sprintf(buf, "out.%d", mpi.rank);
		g_out_file = fopen(buf, "w");
		if(g_out_file == NULL) {
			return stderr;
		}
	}
	fflush(g_out_file);
	return g_out_file;
}

void close_imd_out_file() {
	if(g_out_file != NULL) {
		fclose(g_out_file);
		g_out_file = NULL;
	}
}

//-------------------------------------------------------------//
// Exception
//-------------------------------------------------------------//

void throw_exception(const char* format, ...) {
	char buf[300];
	va_list arg;
	va_start(arg, format);
    vsnprintf(buf, sizeof(buf), format, arg);
    va_end(arg);
#if PRINT_WITH_TIME
    fprintf(IMD_OUT, "[r:%d,%f] %s\n", mpi.rank, global_clock.get() / 1000000.0, buf);
#else
    fprintf(IMD_OUT, "[r:%d] %s\n", mpi.rank, buf);
#endif
    throw buf;
}

void print_prefix() {
#if PRINT_WITH_TIME
	fprintf(IMD_OUT, "[r:%d,%f] ", mpi.rank, global_clock.get() / 1000000.0);
#else
    fprintf(IMD_OUT, "[r:%d] ", mpi.rank, buf);
#endif
}

void print_with_prefix(const char* format, ...) {
	char buf[300];
	va_list arg;
	va_start(arg, format);
    vsnprintf(buf, sizeof(buf), format, arg);
    va_end(arg);
#if PRINT_WITH_TIME
	fprintf(IMD_OUT, "[r:%d,%f] %s\n", mpi.rank, global_clock.get() / 1000000.0, buf);
#else
    fprintf(IMD_OUT, "[r:%d] %s\n", mpi.rank, buf);
#endif
}

//-------------------------------------------------------------//
// Memory Allocation
//-------------------------------------------------------------//

////
int64_t g_memory_usage = 0;
void x_allocate_check(void* ptr) {
	size_t nbytes = malloc_usable_size(ptr);
	g_memory_usage += nbytes;
	if(mpi.isMaster() && nbytes > 1024*1024) {
		fprintf(IMD_OUT, "[MEM] %f MB (+ %f MB)\n", (double)g_memory_usage / (1024*1024), (double)nbytes / (1024*1024));
	}
}
void x_free_check(void* ptr) {
	size_t nbytes = malloc_usable_size(ptr);
	g_memory_usage -= nbytes;
	if(mpi.isMaster() && nbytes > 1024*1024) {
		fprintf(IMD_OUT, "[MEM] %f MB (- %f MB)\n", (double)g_memory_usage / (1024*1024), (double)nbytes / (1024*1024));
	}
}
void print_max_memory_usage() {
	int64_t g_max = 0;
	MPI_Reduce(&g_memory_usage, &g_max, 1, MpiTypeOf<int64_t>::type, MPI_MAX, 0, mpi.comm_2d);
	if(mpi.isMaster()) {
		fprintf(IMD_OUT, "[MEM-MAX] %f MB\n", (double)g_max / (1024*1024));
	}
}
////

void* xMPI_Alloc_mem(size_t nbytes) {
  void* p = NULL;
  MPI_Alloc_mem(nbytes, MPI_INFO_NULL, &p);
  if (nbytes != 0 && !p) {
	  throw_exception("MPI_Alloc_mem failed for size%zu (%"PRId64") byte(s)", nbytes, (int64_t)nbytes);
  }
#if VERVOSE_MODE
  if(mpi.isMaster() && nbytes > 1024*1024) {
    fprintf(IMD_OUT, "[MEM-MPI] + %f MB\n", (double)nbytes / (1024*1024));
  }
#endif
  return p;
}

void* cache_aligned_xcalloc(const size_t size) {
    void* p = NULL;
	if(posix_memalign(&p, CACHE_LINE, size)){
		throw_exception("Out of memory trying to allocate %zu (%"PRId64") byte(s)", size, (int64_t)size);
	}
	VERVOSE(x_allocate_check(p));
	memset(p, 0, size);
	return p;
}
void* cache_aligned_xmalloc(const size_t size) {
	void* p = NULL;
	if(posix_memalign(&p, CACHE_LINE, size)){
		throw_exception("Out of memory trying to allocate %zu (%"PRId64") byte(s)", size, (int64_t)size);
	}
	VERVOSE(x_allocate_check(p));
	return p;
}

void* page_aligned_xcalloc(const size_t size) {
	void* p = NULL;
	if(posix_memalign(&p, PAGE_SIZE, size)){
		throw_exception("Out of memory trying to allocate %zu (%"PRId64") byte(s)", size, (int64_t)size);
	}
	VERVOSE(x_allocate_check(p));
	memset(p, 0, size);
	return p;
}
void* page_aligned_xmalloc(const size_t size) {
	void* p = NULL;
	if(posix_memalign(&p, PAGE_SIZE, size)){
		throw_exception("Out of memory trying to allocate %zu (%"PRId64") byte(s)", size, (int64_t)size);
	}
	VERVOSE(x_allocate_check(p));
	return p;
}

#if VERVOSE_MODE

void xfree(void* p) {
	x_free_check(p);
	free(p);
}
#define free(p) xfree(p)

#endif // #if VERVOSE_MODE

#if SHARED_MEMORY
void* shared_malloc(size_t nbytes) {
	MPI_Comm comm = mpi.comm_z;
	int rank; MPI_Comm_rank(comm, &rank);
	key_t shm_key;
	int shmid = -1;
	void* addr = NULL;

	if(rank == 0) {
		timeval tv; gettimeofday(&tv, NULL);
		shm_key = tv.tv_usec;
		for(int i = 0; i < 1000; ++i) {
			shmid = shmget(++shm_key, nbytes,
					IPC_CREAT | IPC_EXCL | 0600);
			if(shmid != -1) break;
#ifndef NDEBUG
			perror("shmget try");
#endif
		}
		if(shmid == -1) {
			perror("shmget");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		addr = shmat(shmid, NULL, 0);
		if(addr == (void*)-1) {
			perror("Shared memory attach failure");
			addr = NULL;
		}
	}

	MPI_Bcast(&shm_key, 1, MpiTypeOf<key_t>::type, 0, comm);

	if(rank != 0) {
		shmid = shmget(shm_key, 0, 0);
		if(shmid == -1) {
			perror("shmget");
		}
		else {
		addr = shmat(shmid, NULL, 0);
			if(addr == (void*)-1) {
				perror("Shared memory attach failure");
				addr = NULL;
			}
		}
	}

	MPI_Barrier(comm);

	if(rank == 0) {
		// release the memory when the last process is detached.
		if(shmctl(shmid, IPC_RMID, NULL) == -1) {
			perror("shmctl(shmid, IPC_RMID, NULL)");
		}
	}
	return addr;
}

void shared_free(void* shm) {
	if(shmdt(shm) == -1) {
		perror("shmdt(shm)");
	}
}

void test_shared_memory() {
	int* mem = (int*)shared_malloc(sizeof(int));
	int ref_val = 0;
	if(mpi.rank_z == 0) {
		*mem = ref_val = mpi.rank;
	}
	MPI_Bcast(&ref_val, 1, MpiTypeOf<int>::type, 0, mpi.comm_z);
	int result = (*mem == ref_val), global_result;
	shared_free(mem);
	MPI_Allreduce(&result, &global_result, 1, MpiTypeOf<int>::type, MPI_LOR, MPI_COMM_WORLD);
	if(global_result == false) {
		if(mpi.isMaster()) print_with_prefix("Shared memory test failed!! Please, check MPI_NUM_NODE.");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
}
#else // #if SHARED_MEMORY
void* shared_malloc(size_t size) {
	return page_aligned_xcalloc(size);
}
void shared_free(void* shm) {
	free(shm);
}
#endif // #if SHARED_MEMORY

//-------------------------------------------------------------//
// Other functions
//-------------------------------------------------------------//

template <typename T> T roundup(T size, T width)
{
	return (size + width - 1) / width * width;
}

template <typename T> T get_blocks(T size, T width)
{
	return (size + width - 1) / width;
}

template <typename T> T roundup_2n(T size, T width)
{
	return (size + width - 1) & -width;
}

template <typename T>
void get_partition(T size, int num_part, int part_idx, T& begin, T& end) {
	T part_size = (size + num_part - 1) / num_part;
	begin = std::min(part_size * part_idx, size);
	end = std::min(begin + part_size, size);
}

// # of partition = num_blks * parts_factor
template <typename T>
void get_partition(T* blk_offset, int num_blks, int parts_per_blk, int part_idx, T& begin, T& end) {
	int blk_idx = part_idx / parts_per_blk;
	T blk_begin = blk_offset[blk_idx];
	T blk_size = blk_offset[blk_idx+1] - blk_begin;
	get_partition(blk_size, parts_per_blk, part_idx - blk_idx * parts_per_blk, begin, end);
	begin += blk_begin;
	end += blk_begin;
}

template <typename T>
void get_partition(int64_t size, T* sorted, int log_blk,
		int num_part, int part_idx, int64_t& begin, int64_t& end)
{
	if(size == 0) {
		begin = end = 0;
		return ;
	}
	get_partition(size, num_part, part_idx, begin, end);
	if(begin != 0) {
		T piv = sorted[begin] >> log_blk;
		while(begin < size && (sorted[begin] >> log_blk) == piv) ++begin;
	}
	if(end != 0) {
		T piv = sorted[end] >> log_blk;
		while(end < size && (sorted[end] >> log_blk) == piv) ++end;
	}
}
/*
template <typename T>
void get_partition(int64_t size, T* sorted, int64_t min_value, int64_t max_value,
		int64_t min_blk_size, int num_part, int part_idx, T*& begin, T*& end)
{
	T blk_size = std::max(min_blk_size, (max_value - min_value + num_part - 1) / num_part);
	int64_t begin_value = min_value + blk_size * part_idx;
	int64_t end_value = begin_value + blk_size;
	begin = std::lower_bound(sorted, sorted + size, begin_value);
	end = std::lower_bound(sorted, sorted + size, end_value);
}
*/

//-------------------------------------------------------------//
// CPU Affinity Setting
//-------------------------------------------------------------//

int g_GpuIndex = -1;

namespace numa {

int num_omp_threads = 2; // # of threads available
int num_bfs_threads = 1; // # of threads for BFS
#if OPENMP_SUB_THREAD
bool is_extra_omp = true;
#else
bool is_extra_omp = false;
#endif
int next_base_thread_id = 0;
int next_thread_id = 1;

__thread int thread_id = -1;

void cpu_set_to_string(cpu_set_t *set, std::string& str, int num_procs) {
	int set_count = 0;
	for(int i = 0; i < num_procs; ++i) {
		if(CPU_ISSET(i, set)) set_count++;
	}
	if(set_count < 5) {
		for(int i = 0; i < num_procs; ++i) {
			if(CPU_ISSET(i, set)) {
				char buf[100];
				sprintf(buf, "%d,", i);
				str += buf;
			}
		}
	}
	else {
		for(int i = 0; i < num_procs; ++i) {
			if(CPU_ISSET(i, set)) str.push_back('x');
			else str.push_back('-');
		}
	}
}

void print_current_binding(const char* message) {
	cpu_set_t set;
	sched_getaffinity(0, sizeof(set), &set);
	int num_procs = sysconf(_SC_NPROCESSORS_CONF);
	std::string str_affinity;
	cpu_set_to_string(&set, str_affinity, num_procs);
	print_with_prefix("th:%d %s -> [%s]", thread_id, message, str_affinity.c_str());
}

void initialize_num_threads() {
	num_omp_threads = omp_get_max_threads();
	const char* bfs_num_threads_str = getenv("BFS_NTHREADS");
	//num_bfs_threads = num_omp_threads - 1;
	num_bfs_threads = num_omp_threads; // currently there are no background thread
	if(bfs_num_threads_str != NULL) {
		num_bfs_threads = atoi(bfs_num_threads_str);
	}
	if(num_bfs_threads == 0) {
		print_with_prefix("BFS_NTHREADS must be greater than 0");
	}
}

class CoreBinding {
public:
	CoreBinding() {
		num_procs = sysconf(_SC_NPROCESSORS_CONF);
	}
	virtual ~CoreBinding() {
	}
	virtual int cpu(int tid) = 0;
	virtual int num_logical_CPUs() = 0;

	int num_procs;
};

class SimpleCoreBinding : public CoreBinding {
public:
	SimpleCoreBinding()
		: CoreBinding()
	{ }

	virtual int cpu(int tid) {
		return (tid % num_procs);
	}
	virtual int num_logical_CPUs() {
		return num_procs;
	}
};

class ManualCoreBinding : public CoreBinding {
public:
	ManualCoreBinding(int* cpuset, int num_avail_procs)
		: CoreBinding()
	{
		avail_procs.insert(avail_procs.begin(), cpuset, cpuset + num_avail_procs);
	}

	virtual int cpu(int tid) {
		return avail_procs[tid];
	}
	virtual int num_logical_CPUs() {
		return avail_procs.size();
	}

	std::vector<int> avail_procs;
};

#if defined(__i386__) || defined(__x86_64__)
typedef struct cpuid_register_t {
	unsigned long eax;
	unsigned long ebx;
	unsigned long ecx;
	unsigned long edx;
} cpuid_register_t;

void cpuid(unsigned long eax, unsigned long ecx, cpuid_register_t *r)
{
    __asm__ volatile (
        "cpuid"
        :"=a"(r->eax), "=b"(r->ebx), "=c"(r->ecx), "=d"(r->edx)
        :"a"(eax), "c"(ecx)
    );
    return;
}

class AutoDetectCoreBinding : public CoreBinding {
public:
	AutoDetectCoreBinding(int numa_node_size, int numa_node_rank)
	: CoreBinding()
	{
		bool ex_apic_supported = false;
		union {
			char ch[13];
			uint32_t reg[3];
		} vendor_sign;
		cpuid_register_t reg;

		logical_CPU_bits = 0;
		core_bits = 0;

		// get vendor signature
		cpuid(0, 0, &reg);
		int max_basic_id = reg.eax;
		vendor_sign.reg[0] = reg.ebx;
		vendor_sign.reg[1] = reg.edx;
		vendor_sign.reg[2] = reg.ecx;
		vendor_sign.ch[12] = 0;

		cpuid(1, 0, &reg);
		int count_bits = num_bits((reg.ebx >> 16) & 0xFF);

		if(memcmp(vendor_sign.ch, "GenuineIntel", 12) == 0) {
			// Intel
			if(max_basic_id >= 0xB) {
				cpuid(0xB, 0, &reg);
				ex_apic_supported = (reg.ebx != 0);
				logical_CPU_bits = reg.eax & 0x1F;
			}
			if(ex_apic_supported) {
				cpuid(0xB, 1, &reg);
				core_bits = (reg.eax & 0x1F) - logical_CPU_bits;
			}
			else if(max_basic_id >= 4) {
				logical_CPU_bits = count_bits;
				cpuid(4, 0, &reg);
				core_bits = (reg.eax >> 26) - count_bits;
			}
			else {
				logical_CPU_bits = count_bits;
			}
		}
		else if(memcmp(vendor_sign.ch, "AuthenticAMD", 12) == 0) {
			// AMD
			cpuid(0x80000000u, 0, &reg);
			if(reg.eax >= 0x80000008u) {
				cpuid(0x80000008u, 0, &reg);
				int tmp_bits = (reg.eax >> 12) & 0xF;
				if(tmp_bits > 0) {
					core_bits = tmp_bits;
				}
				else {
					core_bits = num_bits(reg.eax & 0xFF);
				}
				logical_CPU_bits = count_bits - core_bits;
			}
			else {
				logical_CPU_bits = count_bits;
			}
		}
		else {
			if(mpi.isMaster()) print_with_prefix("Error: Unknown CPU: %s", vendor_sign.ch);
		}

		cpu_set_t set;
		int32_t core_list[num_procs];
		std::fill(core_list, core_list + num_procs, INT32_MAX);
		int max_apic_id = 0;
		for(int i = 0; i < num_procs; i++) {
			CPU_ZERO(&set);
			CPU_SET(i, &set);
			sched_setaffinity(0, sizeof(set), &set);
			cpu_set_t get_aff;
			sched_getaffinity(0, sizeof(get_aff), &get_aff);
			if(memcmp(&set, &get_aff, sizeof(set))) {
				// skip disabled core
				continue;
			}
			sleep(0);
			cpuid_register_t reg;
			int apicid;
			if(ex_apic_supported) {
				cpuid(0xB, 0, &reg);
				apicid = reg.edx;
			}
			else {
				cpuid(1, 0, &reg);
				apicid = (reg.ebx >> 24) & 0xFF;
			}
			if(max_apic_id < apicid) max_apic_id = apicid;
		//	print_with_prefix("%d-th -> apicid=%d", i, apicid);
			core_list[i] = (apicid << 16) | i;
		}

		std::sort(core_list, core_list + num_procs);
		if(mpi.isMaster()) {
			// print detected numa rank
			print_prefix();
			fprintf(IMD_OUT, "Core list:[");
			for(int i = 0; i < num_procs; i++) {
				if((i % 100) == 0) fprintf(IMD_OUT, "\n%d-%d: ", i, i+100);
				if(core_list[i] != INT32_MAX) {
					fprintf(IMD_OUT, "%d-%d,", core_list[i] >> 16, core_list[i] & 0xFFFF);
				}
			}
			fprintf(IMD_OUT, "\n]\n");
		}
		num_logical_CPUs_within_core = std::lower_bound(core_list, core_list + num_procs, core_list[0] + (1 << (logical_CPU_bits + 16))) - core_list;
		int logical_CPUs_within_numa = std::lower_bound(core_list, core_list + num_procs, core_list[0] + (1 << (core_bits + logical_CPU_bits + 16))) - core_list;
		num_cores_within_numa = logical_CPUs_within_numa / num_logical_CPUs_within_core;
		num_numa_nodes = num_procs / logical_CPUs_within_numa;

		if(num_procs != (num_logical_CPUs_within_core * num_cores_within_numa * num_numa_nodes)) {
			if(mpi.isMaster()) print_with_prefix("Error: Affinity feature does not support heterogeneous systems."
					"(num_procs=%d, SMT=%d, Cores=%d, NUMA Nodes=%d)\n",
					num_procs, num_logical_CPUs_within_core, num_cores_within_numa, num_numa_nodes);
		}

		apicid_to_cpu = new int[max_apic_id + 1]();
		for(int numa = 0; numa < num_numa_nodes; ++numa) {
			for(int core = 0; core < num_cores_within_numa; ++core) {
				for(int smt = 0; smt < num_logical_CPUs_within_core; ++smt) {
					int cpu_idx = ((numa * num_cores_within_numa) + core) * num_logical_CPUs_within_core + smt;
					assert (my_cpu_id(numa, core, smt) <= max_apic_id);
					apicid_to_cpu[my_cpu_id(numa, core, smt)] = (core_list[cpu_idx] & 0xFFFF);
				}
			}
		}

		if(mpi.isMaster()) {
			print_with_prefix("CPU Topology: # of socket: %d, # of cores: %d, # of SMT: %d",
					num_numa_nodes, num_cores_within_numa, num_logical_CPUs_within_core);
/*
			// print detected numa rank
			fprintf(IMD_OUT, "NUMA Rank:[");
			for(int i = 0; i < num_procs; i++) {
				if((i % 100) == 0) fprintf(IMD_OUT, "\n%d-%d: ", i, i+100);
				fprintf(IMD_OUT, "%d,", numa_node_of_cpu(i));
			}
			fprintf(IMD_OUT, "\n]\n");
			*/
		}

		this->numa_node_rank = numa_node_rank;
		procs_per_numa_node = (numa_node_size + num_numa_nodes - 1) / num_numa_nodes;
		idx_within_numa_node = numa_node_rank / num_numa_nodes;
		num_cores_assgined = num_cores_within_numa / procs_per_numa_node;

		if(num_cores_within_numa != procs_per_numa_node * num_cores_assgined) {
			if(mpi.isRmaster()) print_with_prefix("Warning: Core affinity is disabled because we cannot determine the # of cores to assign.");
			disable_affinity = true;
		}
	}

	virtual ~AutoDetectCoreBinding() {
		if(apicid_to_cpu) { delete [] apicid_to_cpu; apicid_to_cpu = NULL; }
	}

	virtual int cpu(int tid) {
		int numa = numa_node_rank % num_numa_nodes;
		int core = (num_cores_assgined * idx_within_numa_node) + (tid % num_cores_assgined);
		int smt = (tid / num_cores_assgined) % num_logical_CPUs_within_core;

		return apicid_to_cpu[my_cpu_id(numa, core, smt)];
	}

	virtual int num_logical_CPUs() { return num_cores_assgined * num_logical_CPUs_within_core; }

	int cpu(int numa_rank, int core_rank, int smt_rank) {
		if(apicid_to_cpu == NULL || numa_rank >= num_numa_nodes || core_rank >= num_cores_within_numa || smt_rank >= num_logical_CPUs_within_core)
			throw_exception("Invalid rank");

		return apicid_to_cpu[my_cpu_id(numa_rank, core_rank, smt_rank)];
	}

	int num_bits(uint32_t count) {
		if(count <= 1) return 0;
		return get_msb_index(count - 1) + 1;
	}

	int my_cpu_id(int numa_rank, int core_rank, int smt_rank) {
		return (((numa_rank << core_bits) | core_rank) << logical_CPU_bits) | smt_rank;
	}

	bool disable_affinity;
	int num_logical_CPUs_within_core;
	int num_cores_within_numa;
	int num_numa_nodes;
	int logical_CPU_bits;
	int core_bits;
	int *apicid_to_cpu;
private:
	int numa_node_rank;
	int procs_per_numa_node;
	int idx_within_numa_node;
	int num_cores_assgined;
};
#endif // #if defined(__i386__) || defined(__x86_64__)

enum AffinityMode {
	SIMPLE_AFFINITY = 0,
	AUTO_DETECT = 1,
	MAPPING_FILE = 2, // Not supported
	USE_EXISTING_AFFINITY = 3,
};

CoreBinding *core_binding = NULL;
#if defined(__i386__) || defined(__x86_64__)
AffinityMode affinity_mode = AUTO_DETECT;
__thread int my_apic_id = -1;

inline int get_apic_id() {
	cpuid_register_t reg;
	cpuid(0xB, 0, &reg);
	return reg.edx;
}

void initialize_core_id() {
	my_apic_id = get_apic_id();
}
/*
bool ensure_my_apic_id() {
	int cur_id = get_apic_id();
	if(my_apic_id != cur_id) {
		throw_exception("Fatal error: Affinity is changed unexpectedly!!!(%d -> %d)\n", my_apic_id, cur_id);
	}
	return false;
}
*/
#else
AffinityMode affinity_mode = USE_EXISTING_AFFINITY;
void initialize_core_id() { }
#endif // #if defined(__i386__) || defined(__x86_64__)
bool core_affinity_enabled = false;

__thread cpu_set_t initial_affinity;

void print_bind_mode() {
	const char* mode_str = "Auto";
	switch(affinity_mode) {
	case SIMPLE_AFFINITY:
		mode_str = "Simple Non-NUMA affinity";
		break;
	case USE_EXISTING_AFFINITY:
		mode_str = "Use existing affinity";
		break;
	case AUTO_DETECT:
	case MAPPING_FILE:
		break;
	}
	print_with_prefix("Core bind mode: %s", mode_str);
}

void check_affinity_setting() {
	if(core_affinity_enabled == false || core_binding == NULL) return ;
	if(thread_id == -1) {
		print_with_prefix("th:%d affinity is not set", thread_id);
		return ;
	}
	cpu_set_t set;
	sched_getaffinity(0, sizeof(set), &set);
	if(memcmp(&set, &initial_affinity, sizeof(set))) {
		std::string init_str, now_str;
		cpu_set_to_string(&initial_affinity, init_str, core_binding->num_procs);
		cpu_set_to_string(&set, now_str, core_binding->num_procs);
		print_with_prefix("th:%d affinity is changed [%s] -> [%s]",
				thread_id, init_str.c_str(), now_str.c_str());
	}
}

void internal_set_core_affinity(int cpu) {
	CPU_ZERO(&initial_affinity);
	CPU_SET(cpu, &initial_affinity);
	sched_setaffinity(0, sizeof(initial_affinity), &initial_affinity);
	sleep(0);
	initialize_core_id();
#if PRINT_BINDING
	std::string str_affinity;
	cpu_set_to_string(&initial_affinity, str_affinity, sysconf(_SC_NPROCESSORS_CONF));
	print_with_prefix("th:%d thread started [%s](%d)", thread_id, str_affinity.c_str(), cpu);
#endif // #if PRINT_BINDING
	check_affinity_setting();
}

void* empty_function(void*) { return NULL; }

void launch_dummy_thread(int num_dummy_threads) {
	for(int i = 0; i < num_dummy_threads; ++i) {
		pthread_t thread;
		pthread_create(&thread, NULL, empty_function, NULL);
		pthread_join(thread, NULL);
	}
}

/**
 * This function obtain the current core binding
 * and returns the cpu set the threads are bound to
 * and also set the thread_id and the number of OpenMP threads
 */
bool detect_core_affinity(std::vector<int>& cpu_set) {
	if(num_bfs_threads > num_omp_threads) {
		print_with_prefix("BFS_NTHREADS must be equal or less than OMP_NUM_THREADS");
		return false;
	}
	//int total_threads = std::max(num_omp_threads, num_bfs_threads+1);
	int total_threads = std::max(num_omp_threads, num_bfs_threads); // currently there are no background thread
	cpu_set.resize(total_threads, 0);
	bool core_affinity = false, process_affinity = false;
	int num_procs = sysconf(_SC_NPROCESSORS_CONF);

#if SGI_OMPLACE_BUG
	launch_dummy_thread(1);
#endif // #if SGI_OMPLACE_BUG

#pragma omp parallel num_threads(num_omp_threads) reduction(|:core_affinity, process_affinity)
	{
		thread_id = omp_get_thread_num();
		cpu_set_t set;
		sched_getaffinity(0, sizeof(set), &set);
		int cnt = 0;
		for(int i = 0; i < num_procs; i++) {
			if(CPU_ISSET(i, &set)) {
				cnt++;
			}
		}
		if(cnt == 1) {
			// Core affinity is set
			core_affinity = true;
#if PRINT_BINDING
			print_current_binding("detected core binding");
#endif
			for(int i = 0; i < num_procs; i++) {
				if(CPU_ISSET(i, &set)) {
					cpu_set[thread_id] = i;
					initialize_core_id();
					break;
				}
			}
		}
		else if(cnt >= total_threads) {
			// Process affinity is set.
			process_affinity = true;
#if PRINT_BINDING
			print_current_binding("detected process binding");
#endif
			if(thread_id == 0) {
				int cpu_idx = 0;
				for(int i = 0; i < num_procs; i++) {
					if(CPU_ISSET(i, &set)) {
						cpu_set[cpu_idx++] = i;
						if(cpu_idx == total_threads) {
							break;
						}
					}
				}
			}
		}
		else {
			// Affinity is set ???
#if PRINT_BINDING
			print_current_binding("??? binding");
#endif
			core_affinity = process_affinity = true;
		}
	}
	if(core_affinity && process_affinity) {
		print_with_prefix("Error: failed to detect existing core binding. hetero ?");
		return false;
	}
	if(process_affinity) {
		if(mpi.isRmaster()) print_with_prefix("Detect process affinity");
#pragma omp parallel num_threads(num_omp_threads)
		internal_set_core_affinity(cpu_set[thread_id]);
	}
	else {
		// add remaining cores
		if(total_threads > num_omp_threads) {
			cpu_set[num_omp_threads] = num_omp_threads;
		}
		if(mpi.isRmaster()) print_with_prefix("Detect core affinity");
	}
	omp_set_num_threads(num_bfs_threads);
	return true;
}

/**
 * affinity setting for master threads.
 */
void set_core_affinity() {
	if(thread_id == -1) {
		thread_id = __sync_fetch_and_add(&next_base_thread_id, 1);
		if(core_binding != NULL) {
			// This code assumes that there are no nested openmp parallel sections.
			if(core_binding->num_logical_CPUs() <= num_omp_threads) {
				print_with_prefix("th:%d Too many threads.", thread_id);
			}
			int core_id = (thread_id % core_binding->num_logical_CPUs());
			if(core_affinity_enabled) {
				int cpu = core_binding->cpu(core_id);
				internal_set_core_affinity(cpu);
			}
		}
		else {
#if PRINT_BINDING
			print_current_binding("started");
#endif
		}
		if((mpi.thread_level == MPI_THREAD_SINGLE) || thread_id == 0) {
			omp_set_num_threads(num_bfs_threads);
		}
	}
#if CPU_BIND_CHECK
	else if(core_affinity_enabled) {
		ensure_my_apic_id();
	}
#endif
}

void set_omp_core_affinity() {
	if(thread_id == -1) {
		if(core_binding != NULL) {
			// This code assumes that there are no nested openmp parallel sections.
			int core_id;
			do {
				thread_id = __sync_fetch_and_add(&next_thread_id, 1);
				core_id = (thread_id % core_binding->num_logical_CPUs());
			} while(core_id == 0 || core_id >= num_bfs_threads+(is_extra_omp?1:0));
			if(core_affinity_enabled) {
				int cpu = core_binding->cpu(core_id);
				internal_set_core_affinity(cpu);
			}
		}
		else {
			thread_id = __sync_fetch_and_add(&next_thread_id, 1);
#if PRINT_BINDING
			print_current_binding("started");
#endif
		}
	}
#if CPU_BIND_CHECK
	else if(core_affinity_enabled) {
		ensure_my_apic_id();
	}
#endif
}

#define SET_AFFINITY numa::set_core_affinity()
#define SET_OMP_AFFINITY numa::set_omp_core_affinity()

void set_affinity()
{
	const char* num_node_str = getenv("MPI_NUM_NODE");
	int num_node;
	if(num_node_str != NULL) {
		num_node = atoi(num_node_str);
	}
	else {
		num_node = mpi.size;
		if(mpi.isRmaster()) {
			print_with_prefix("Warning: failed to get # of node (MPI_NUM_NODE=<# of node>). We assume 1 process per node");
		}
	}
	const char* dist_round_robin = getenv("MPI_ROUND_ROBIN");
	int max_procs_per_node = (mpi.size + num_node - 1) / num_node;
	int proc_rank = (dist_round_robin ? (mpi.rank / num_node) : (mpi.rank % max_procs_per_node));
	g_GpuIndex = proc_rank;

	if(mpi.isRmaster()) {
		print_with_prefix("process distribution : %s", dist_round_robin ? "round robin" : "partition");
	}
#if SHARED_MEMORY
	if(max_procs_per_node > 1 && max_procs_per_node != 3) {
		mpi.size_z = max_procs_per_node;
		mpi.rank_z = proc_rank;

		// create comm_z
		if(mpi.size_z > 1) {
			if(dist_round_robin) {
				MPI_Comm_split(MPI_COMM_WORLD, mpi.rank % num_node, mpi.rank_z, &mpi.comm_z);
			}
			else {
				MPI_Comm_split(MPI_COMM_WORLD, mpi.rank / max_procs_per_node, mpi.rank_z, &mpi.comm_z);
			}

			// test shared memory
			test_shared_memory();

			// create comm_y
			if(dist_round_robin == false && mpi.isRowMajor == false) {
				mpi.rank_y = mpi.rank_2dc / mpi.size_z;
				mpi.size_y = mpi.size_2dr / mpi.size_z;
				MPI_Comm_split(mpi.comm_2dc, mpi.rank_z, mpi.rank_2dc / mpi.size_z, &mpi.comm_y);
			}
		}
	}
#endif
	const char* core_bind = getenv("CORE_BIND");
	if(core_bind != NULL) {
		affinity_mode = (AffinityMode)atoi(core_bind);
	}
	if(mpi.isRmaster()) print_bind_mode();
	if(affinity_mode == USE_EXISTING_AFFINITY) {
		std::vector<int> cpu_set;
		if(detect_core_affinity(cpu_set) == false) {
			affinity_mode = SIMPLE_AFFINITY;
		}
		else {
			core_affinity_enabled = true;
			core_binding = new ManualCoreBinding(&cpu_set[0], cpu_set.size());
			if(mpi.isRmaster()) print_with_prefix("Core affinity is enabled (using existing affinigy)");
		}
	}
	if(affinity_mode == SIMPLE_AFFINITY) {
		core_affinity_enabled = true;
		core_binding = new SimpleCoreBinding();
	}
#if NUMA_BIND
	if(core_binding == NULL) {
		AutoDetectCoreBinding* topology = new AutoDetectCoreBinding(max_procs_per_node, proc_rank);
		if(max_procs_per_node > 1) {
			if(max_procs_per_node == 3) {
				if(numa_available() < 0) {
					print_with_prefix("No NUMA support available on this system.");
				}
				else {
					int NUM_SOCKET = numa_max_node() + 1;
					if(proc_rank < NUM_SOCKET) {
						numa_set_preferred(proc_rank);
						numa_run_on_node(proc_rank);
					}
					else {
						cpu_set_t set; CPU_ZERO(&set);
						for(int i = 0; i < topology->num_procs; i++) {
							CPU_SET(i, &set);
						}
						sched_setaffinity(0, sizeof(set), &set);
					}
					if(NUM_SOCKET != topology->num_numa_nodes) {
						if(mpi.isMaster()) print_with_prefix("Warning: # of NUMA nodes from the libnuma does not match ours. (libnuma = %d, ours = %d)",
								NUM_SOCKET, topology->num_numa_nodes);
					}
				}
				/*
				cpu_set_t set; CPU_ZERO(&set);
				if(proc_rank < topology->num_numa_nodes) {
					for(int core = 0; core < topology->num_cores_within_numa; core++)
						for(int smt = 0; smt < topology->num_cores_within_numa; smt++)
							CPU_SET(topology->cpu(proc_rank, core, smt), &set);
				}
				else {
					for(int i = 0; i < topology->num_procs; i++)
						CPU_SET(i, &set);
				}
				sched_setaffinity(0, sizeof(set), &set);
				*/
				// disable core binding
				delete topology; topology = NULL;

				if(mpi.isRmaster()) { /* print from max rank node for easy debugging */
				  print_with_prefix("affinity for executing 3 processed per node is enabled.");
				}
			}
			else {
				if(numa_available() < 0) {
					print_with_prefix("No NUMA support available on this system.");
					return ;
				}
				int NUM_SOCKET = numa_max_node() + 1;
				if(NUM_SOCKET != topology->num_numa_nodes) {
					if(mpi.isRmaster()) print_with_prefix("Warning: # of NUMA nodes from the libnuma does not match ours. (libnuma = %d, ours = %d)",
							NUM_SOCKET, topology->num_numa_nodes);
				}

				// set default affinity to numa node
				numa_run_on_node(proc_rank % NUM_SOCKET);

				// memory affinity
				numa_set_preferred(proc_rank % NUM_SOCKET);

				core_affinity_enabled = true;
				if(mpi.isRmaster()) print_with_prefix("Core affinity is enabled");

				if(mpi.isRmaster()) { /* print from max rank node for easy debugging */
				  print_with_prefix("NUMA node affinity is enabled.");
				}
			}
		}
		// failed to detect CPU topology or there is only one process here
		else {
			if(mpi.isRmaster()) { /* print from max rank node for easy debugging */
			  print_with_prefix("affinity is disabled.");
			}
			cpu_set_t set; CPU_ZERO(&set);
			for(int i = 0; i < topology->num_procs; i++) {
				CPU_SET(i, &set);
			}
			sched_setaffinity(0, sizeof(set), &set);
		}
		core_binding = topology;
	}
#endif // #if NUMA_BIND
	if(mpi.isMaster()) {
		  print_with_prefix("Y dimension is %s", mpi.isYdimAvailable() ? "Enabled" : "Disabled");
	}
	// set main thread's affinity
	set_core_affinity();
	next_base_thread_id = num_bfs_threads+(is_extra_omp?1:0);
	next_thread_id = 1;
}

} // namespace numa

//-------------------------------------------------------------//
// ?
//-------------------------------------------------------------//

/**
 * compute rank that is assigned continuously in the field
 * the last dimension size should be even.
 */
static void compute_rank(std::vector<int>& ss, std::vector<int>& rs, COMM_2D& c) {
	int size = 1;
	int rank = 0;
	for(int i = ss.size() - 1; i >= 0; --i) {
		if(rank % 2) {
			rank = rank * ss[i] + (ss[i] - 1 - rs[i]);
		}
		else {
			rank = rank * ss[i] + rs[i];
		}
		size *= ss[i];
	}
	c.size_x = ss[0];
	c.size_y = size / c.size_x;
	c.rank_x = rs[0];
	c.rank_y = rank / c.size_x;
	c.rank = rank;
	c.size = size;
}

static int compute_rank_2d(int x, int y, int sx, int sy) {
	if(x >= sx) x -= sx;
	if(x < 0) x += sx;
	if(y >= sy) y -= sy;
	if(y < 0) y += sy;

	if(y % 2)
		return y * sx + (sx - 1 - x);
	else
		return y * sx + x;
}

static void setup_rank_map(COMM_2D& comm) {
	int send_rank = comm.rank_x + comm.rank_y * comm.size_x;
	int recv_rank[comm.size];
	MPI_Allgather(&send_rank, 1, MPI_INT, recv_rank, 1, MPI_INT, comm.comm);
	comm.rank_map = (int*)malloc(comm.size*sizeof(int));
	for(int i = 0; i < comm.size; ++i) {
		comm.rank_map[recv_rank[i]] = i;
	}
}

#if ENABLE_FJMPI
static void parse_row_dims(bool* rdim, const char* input) {
	memset(rdim, 0x00, sizeof(bool)*6);
	while(*input) {
		switch(*(input++)) {
		case 'x':
			rdim[0] = true;
			break;
		case 'y':
			rdim[1] = true;
			break;
		case 'z':
			rdim[2] = true;
			break;
		case 'a':
			rdim[3] = true;
			break;
		case 'b':
			rdim[4] = true;
			break;
		case 'c':
			rdim[5] = true;
			break;
		}
	}
}

static void print_dims(const char* prefix, std::vector<int>& dims) {
	print_prefix();
	fprintf(IMD_OUT, "%s%d", prefix, dims[0]);
	for(int i = 1; i < int(dims.size()); ++i) {
		fprintf(IMD_OUT, "x%d", dims[i]);
	}
	fprintf(IMD_OUT, "\n");
}
#endif

static void setup_2dcomm()
{
	bool success = false;
	mpi.isMultiDimAvailable = false;

#if ENABLE_FJMPI
	const char* tofu_6d = getenv("TOFU_6D");
	if(!success && tofu_6d) {
		int rank6d[6];
		int size6d[6];
		FJMPI_Topology_rel_rank2xyzabc(mpi.rank, &rank6d[0], &rank6d[1], &rank6d[2], &rank6d[3], &rank6d[4], &rank6d[5]);
		MPI_Allreduce(rank6d, size6d, 6, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		int total = 1;
		for(int i = 0; i < 6; ++i) {
			total *= ++size6d[i];
		}
		if(mpi.isMaster()) print_with_prefix("Detected dimension %dx%dx%dx%dx%dx%d = %d", size6d[0], size6d[1], size6d[2], size6d[3], size6d[4], size6d[5], mpi.size);
		if(total != mpi.size) {
			if(mpi.isMaster()) print_with_prefix("Mismatch error!");
		}
		else {
			bool rdim[6] = {0};
			parse_row_dims(rdim, tofu_6d);
			std::vector<int> ss_r, rs_r;
			std::vector<int> ss_c, rs_c;
			for(int i = 0; i < 6; ++i) {
				if(rdim[i]) {
					ss_r.push_back(size6d[i]);
					rs_r.push_back(rank6d[i]);
				}
				else {
					ss_c.push_back(size6d[i]);
					rs_c.push_back(rank6d[i]);
				}
			}
			compute_rank(ss_c, rs_c, mpi.comm_r);
			if(mpi.isMaster()) print_dims("R: ", ss_r);
			compute_rank(ss_r, rs_r, mpi.comm_c);
			if(mpi.isMaster()) print_dims("C: ", ss_c);

			mpi.size_2dr = mpi.comm_c.size;
			mpi.size_2dc = mpi.comm_r.size;
			mpi.rank_2dr = mpi.comm_c.rank;
			mpi.rank_2dc = mpi.comm_r.rank;
			mpi.isMultiDimAvailable = true;

			//print_with_prefix("rank: (%d,%d,%d,%d,%d,%d) -> (%d,%d)",
			//		rank6d[0], rank6d[1], rank6d[2], rank6d[3], rank6d[4], rank6d[5], mpi.rank_2dr, mpi.rank_2dc);

			success = true;
		}
	}
#endif // #if ENABLE_FJMPI

	const char* virt_4d = getenv("VIRT_4D");
	if(!success && virt_4d) {
		int RX, RY, CX, CY;
		sscanf(virt_4d, "%dx%dx%dx%d", &RX, &RY, &CX, &CY);
		if(mpi.isMaster()) print_with_prefix("Provided dimension (RXxRYxCXxCY) = %dx%dx%dx%d = %d", RX, RY, CX, CY, mpi.size);
		if(RX*RY*CX*CY != mpi.size) {
			if(mpi.isMaster()) print_with_prefix("Mismatch error!");
		}

		int psr = RX * RY;
		int pr = mpi.rank % psr;
		int pc = mpi.rank / psr;

		std::vector<int> ss, rs;
		ss.push_back(RX);
		ss.push_back(RY);
		rs.push_back(pr % RX);
		rs.push_back(pr / RX);
		compute_rank(ss, rs, mpi.comm_c);
		ss.clear(); rs.clear();

		ss.push_back(CX);
		ss.push_back(CY);
		rs.push_back(pc % CX);
		rs.push_back(pc / CX);
		compute_rank(ss, rs, mpi.comm_r);

		mpi.size_2dr = mpi.comm_c.size;
		mpi.size_2dc = mpi.comm_r.size;
		mpi.rank_2dr = mpi.comm_c.rank;
		mpi.rank_2dc = mpi.comm_r.rank;
		mpi.isMultiDimAvailable = true;

		success = true;
	}

	if(!success) {
		int twod_r = 1, twod_c = 1;
		const char* twod_r_str = getenv("TWOD_R");
		if(twod_r_str){
			twod_r = atoi((char*)twod_r_str);
			twod_c = mpi.size / twod_r;
			if(twod_r == 0 || (twod_c * twod_r) != mpi.size) {
				if(mpi.isMaster()) print_with_prefix("# of MPI processes(%d) cannot be divided by %d", mpi.size, twod_r);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		else {
			for(twod_r = (int)sqrt(mpi.size); twod_r < mpi.size; ++twod_r) {
				twod_c = mpi.size / twod_r;
				if(twod_c * twod_r == mpi.size) {
					break;
				}
			}
			if(twod_c * twod_r != mpi.size) {
				if(mpi.isMaster()) print_with_prefix("Could not find the RxC combination for the # of processes(%d)", mpi.size);
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}

		mpi.comm_c.size = mpi.size_2dr = twod_r;
		mpi.comm_r.size = mpi.size_2dc = twod_c;
		mpi.comm_c.rank = mpi.rank_2dr = mpi.rank % mpi.size_2dr;
		mpi.comm_r.rank = mpi.rank_2dc = mpi.rank / mpi.size_2dr;
	}

	if(mpi.isMaster()) print_with_prefix("Dimension: (%dx%d)", mpi.size_2dr, mpi.size_2dc);

	mpi.isRowMajor = false;
	if(getenv("INVERT_RC")) {
		mpi.isRowMajor = true;
		std::swap(mpi.size_2dr, mpi.size_2dc);
		std::swap(mpi.rank_2dr, mpi.rank_2dc);
		swap(mpi.comm_r, mpi.comm_c);
		if(mpi.isMaster()) print_with_prefix("Inverted: (%dx%d)", mpi.size_2dr, mpi.size_2dc);
	}

	mpi.rank_2d = mpi.rank_2dr + mpi.rank_2dc * mpi.size_2dr;
	mpi.size_2d = mpi.size_2dr * mpi.size_2dc;
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dc, mpi.rank_2dr, &mpi.comm_2dc);
	mpi.comm_c.comm = mpi.comm_2dc;
	MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dr, mpi.rank_2dc, &mpi.comm_2dr);
	mpi.comm_r.comm = mpi.comm_2dr;
	MPI_Comm_split(MPI_COMM_WORLD, 0, mpi.rank_2d, &mpi.comm_2d);

	if(mpi.isMultiDimAvailable) {
		setup_rank_map(mpi.comm_r);
		setup_rank_map(mpi.comm_c);
	}
}

// assume rank = XYZ
static void setup_2dcomm_on_3d()
{
	const char* treed_map_str = getenv("THREED_MAP");
	if(treed_map_str) {
		int X, Y, Z1, Z2;
		sscanf(treed_map_str, "%dx%dx%dx%d", &X, &Y, &Z1, &Z2);
		mpi.size_2dr = X * Z1;
		mpi.size_2dc = Y * Z2;

		if(mpi.isMaster()) fprintf(IMD_OUT, "Dimension: (%dx%dx%dx%d) -> (%dx%d)\n", X, Y, Z1, Z2, mpi.size_2dr, mpi.size_2dc);
		if(mpi.size != mpi.size_2dr * mpi.size_2dc) {
			if(mpi.isMaster()) fprintf(IMD_OUT, "Error: # of processes does not match\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		int x, y, z1, z2;
		x = mpi.rank % X;
		y = (mpi.rank / X) % Y;
		z1 = (mpi.rank / (X*Y)) % Z1;
		z2 = mpi.rank / (X*Y*Z1);
		mpi.rank_2dr = z1 * X + x;
		mpi.rank_2dc = z2 * Y + y;

		mpi.rank_2d = mpi.rank_2dr + mpi.rank_2dc * mpi.size_2dr;
		mpi.size_2d = mpi.size_2dr * mpi.size_2dc;
		MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dc, mpi.rank_2dr, &mpi.comm_2dc);
		MPI_Comm_split(MPI_COMM_WORLD, mpi.rank_2dr, mpi.rank_2dc, &mpi.comm_2dr);
		MPI_Comm_split(MPI_COMM_WORLD, 0, mpi.rank_2d, &mpi.comm_2d);
	}
	else {
		if(mpi.isMaster()) fprintf(IMD_OUT, "Program error.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

}

void cleanup_2dcomm()
{
	if(mpi.isMultiDimAvailable) {
		free(mpi.comm_r.rank_map);
		free(mpi.comm_c.rank_map);
	}
	MPI_Comm_free(&mpi.comm_2dr);
	MPI_Comm_free(&mpi.comm_2dc);
	close_imd_out_file();
}

void setup_globals(int argc, char** argv, int SCALE, int edgefactor)
{
#if BACKTRACE_ON_SIGNAL
	{ // block PRINT_BT_SIGNAL so that only dedicated thread receive the signal
		sigset_t set;
		sigemptyset(&set);
		sigaddset(&set, PRINT_BT_SIGNAL);
		int s = pthread_sigmask(SIG_BLOCK, &set, NULL);
		if(s != 0) throw_exception("failed to set sigmask");
	}
#endif
#if MPI_FUNNELED
	int reqeust_level = MPI_THREAD_FUNNELED;
#else
	int reqeust_level = MPI_THREAD_SINGLE;
#endif
	MPI_Init_thread(&argc, &argv, reqeust_level, &mpi.thread_level);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
#if ENABLE_FJMPI_RDMA
	FJMPI_Rdma_init();
#endif
#if PRINT_WITH_TIME
	global_clock.init();
#endif

	const char* prov_str = "unknown";
	switch(mpi.thread_level) {
	case MPI_THREAD_SINGLE:
		prov_str = "MPI_THREAD_SINGLE";
		break;
	case MPI_THREAD_FUNNELED:
		prov_str = "MPI_THREAD_FUNNELED";
		break;
	case MPI_THREAD_SERIALIZED:
		prov_str = "MPI_THREAD_SERIALIZED";
		break;
	case MPI_THREAD_MULTIPLE:
		prov_str = "MPI_THREAD_MULTIPLE";
		break;
	}

	if(mpi.isMaster()) {
		print_with_prefix("Graph500 Benchmark: SCALE: %d, edgefactor: %d %s", SCALE, edgefactor,
#ifdef NDEBUG
				""
#else
				"(Debug Mode)"
#endif
		);
		print_with_prefix("Running Binary: %s", argv[0]);
		print_with_prefix("Provided MPI thread mode: %s", prov_str);
		print_with_prefix("Pre running time will be %d seconds", PRE_EXEC_TIME);
#if PRINT_WITH_TIME
		char buf[200];
		strftime(buf, sizeof(buf), "%Y/%m/%d %A %H:%M:%S %Z", localtime(&global_clock.l.tv_sec));
		print_with_prefix("Clock started at %s\n", buf);
#endif
	}

#if BACKTRACE_ON_SIGNAL
	backtrace::start_thread();
#endif

#if OPENMP_SUB_THREAD
	omp_set_nested(1);
#endif

	if(getenv("THREED_MAP")) {
		setup_2dcomm_on_3d();
	}
	else {
		setup_2dcomm();
	}

	// Initialize comm_[yz]
	mpi.comm_y = mpi.comm_2dc;
	mpi.comm_z = MPI_COMM_SELF;
	mpi.size_y = mpi.size_2dr;
	mpi.size_z = 1;
	mpi.rank_y = mpi.rank_2dr;
	mpi.rank_z = 0;

	// change default error handler
	MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);

#ifdef _OPENMP
	if(mpi.isRmaster()){
#if _OPENMP >= 200805
	  omp_sched_t kind;
	  int modifier;
	  omp_get_schedule(&kind, &modifier);
	  const char* kind_str = "unknown";
	  switch(kind) {
		case omp_sched_static:
		  kind_str = "omp_sched_static";
		  break;
		case omp_sched_dynamic:
		  kind_str = "omp_sched_dynamic";
		  break;
		case omp_sched_guided:
		  kind_str = "omp_sched_guided";
		  break;
		case omp_sched_auto:
		  kind_str = "omp_sched_auto";
		  break;
	  }
	  print_with_prefix("OpenMP default scheduling : %s, %d", kind_str, modifier);
#else
	  print_with_prefix("OpenMP version : %d", _OPENMP);
#endif
	}
#endif

	UnweightedEdge::initialize();
	UnweightedPackedEdge::initialize();
	WeightedEdge::initialize();

	// check page size
	if(mpi.isMaster()) {
		long page_size = sysconf(_SC_PAGESIZE);
		if(page_size != PAGE_SIZE) {
			print_with_prefix("System Page Size: %ld", page_size);
			print_with_prefix("Error: PAGE_SIZE(%d) is not correct.", PAGE_SIZE);
		}
	}

	// set affinity
	numa::initialize_num_threads();
	if(getenv("NO_AFFINITY") == NULL) {
		numa::set_affinity();
	}

#if CUDA_ENABLED
	CudaStreamManager::initialize_cuda(g_GpuIndex);

	MPI_INFO_ON_GPU mpig;
	mpig.rank = mpi.rank;
	mpig.size = mpi.size_;
	mpig.rank_2d = mpi.rank_2d;
	mpig.rank_2dr = mpi.rank_2dr;
	mpig.rank_2dc = mpi.rank_2dc;
	CudaStreamManager::begin_cuda();
	CUDA_CHECK(cudaMemcpyToSymbol("mpig", &mpig, sizeof(mpig), 0, cudaMemcpyHostToDevice));
	CudaStreamManager::end_cuda();
#endif
}

void cleanup_globals()
{
#if NUMA_BIND
#pragma omp parallel
	numa::check_affinity_setting();
#endif

	cleanup_2dcomm();

	UnweightedEdge::uninitialize();
	UnweightedPackedEdge::uninitialize();
	WeightedEdge::uninitialize();

#if CUDA_ENABLED
	CudaStreamManager::finalize_cuda();
#endif
#if BACKTRACE_ON_SIGNAL
	backtrace::thread_join();
#endif
#if ENABLE_FJMPI_RDMA
	FJMPI_Rdma_finalize();
#endif
	MPI_Finalize();
}

//-------------------------------------------------------------//
// MPI helper
//-------------------------------------------------------------//

namespace MpiCol {

template <typename T>
int allgatherv(T* sendbuf, T* recvbuf, int sendcount, MPI_Comm comm, int comm_size) {
	TRACER(MpiCol::allgatherv);
	int recv_off[comm_size+1], recv_cnt[comm_size];
	MPI_Allgather(&sendcount, 1, MPI_INT, recv_cnt, 1, MPI_INT, comm);
	recv_off[0] = 0;
	for(int i = 0; i < comm_size; ++i) {
		recv_off[i+1] = recv_off[i] + recv_cnt[i];
	}
	MPI_Allgatherv(sendbuf, sendcount, MpiTypeOf<T>::type,
			recvbuf, recv_cnt, recv_off, MpiTypeOf<T>::type, comm);
	return recv_off[comm_size];
}

template <typename T>
void alltoall(T* sendbuf, T* recvbuf, int sendcount, MPI_Comm comm) {
	MPI_Alltoall(sendbuf, sendcount, MpiTypeOf<T>::type,
			recvbuf, sendcount, MpiTypeOf<T>::type, comm);
}

/**
 * @param sendbuf [in]
 * @param sendcount [in]
 * @param sendoffset [out]
 * @param recvcount [out]
 * @param recvoffset [out]
 */
template <typename T>
T* alltoallv(T* sendbuf, int* sendcount,
		int* sendoffset, int* recvcount, int* recvoffset, MPI_Comm comm, int comm_size)
{
	sendoffset[0] = 0;
	for(int r = 0; r < comm_size; ++r) {
		sendoffset[r + 1] = sendoffset[r] + sendcount[r];
	}
	MPI_Alltoall(sendcount, 1, MPI_INT, recvcount, 1, MPI_INT, comm);
	// calculate offsets
	recvoffset[0] = 0;
	for(int r = 0; r < comm_size; ++r) {
		recvoffset[r + 1] = recvoffset[r] + recvcount[r];
	}
	T* recv_data = static_cast<T*>(xMPI_Alloc_mem(recvoffset[comm_size] * sizeof(T)));
	MPI_Alltoallv(sendbuf, sendcount, sendoffset, MpiTypeOf<T>::type,
			recv_data, recvcount, recvoffset, MpiTypeOf<T>::type, comm);
	return recv_data;
}

template <typename T>
void my_allgatherv(T *buffer, int* count, int* offset, MPI_Comm comm, int rank, int size, int left, int right)
{
	int l_sendidx = rank;
	int l_recvidx = (rank + size + 1) % size;
	int r_sendidx = rank;
	int r_recvidx = (rank + size - 1) % size;

	for(int i = 1; i < size; ++i, ++l_sendidx, ++l_recvidx, --r_sendidx, --r_recvidx) {
		if(l_sendidx >= size) l_sendidx -= size;
		if(l_recvidx >= size) l_recvidx -= size;
		if(r_sendidx < 0) r_sendidx += size;
		if(r_recvidx < 0) r_recvidx += size;

		int l_send_off = offset[l_sendidx];
		int l_send_cnt = count[l_sendidx] / 2;
		int l_recv_off = offset[l_recvidx];
		int l_recv_cnt = count[l_recvidx] / 2;

		int r_send_off = offset[r_sendidx] + count[r_sendidx] / 2;
		int r_send_cnt = count[r_sendidx] - count[r_sendidx] / 2;
		int r_recv_off = offset[r_recvidx] + count[r_recvidx] / 2;
		int r_recv_cnt = count[r_recvidx] - count[r_recvidx] / 2;

		MPI_Request req[4];
		MPI_Irecv(&buffer[l_recv_off], l_recv_cnt, MpiTypeOf<T>::type, right, PRM::MY_EXPAND_TAG1, comm, &req[2]);
		MPI_Irecv(&buffer[r_recv_off], r_recv_cnt, MpiTypeOf<T>::type, left, PRM::MY_EXPAND_TAG1, comm, &req[3]);
		MPI_Isend(&buffer[l_send_off], l_send_cnt, MpiTypeOf<T>::type, left, PRM::MY_EXPAND_TAG1, comm, &req[0]);
		MPI_Isend(&buffer[r_send_off], r_send_cnt, MpiTypeOf<T>::type, right, PRM::MY_EXPAND_TAG1, comm, &req[1]);
		MPI_Waitall(4, req, MPI_STATUS_IGNORE);
	}
}

#if 0
template <typename T>
void my_allgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm)
{
	int size; MPI_Comm_size(comm, &size);
	int rank; MPI_Comm_rank(comm, &rank);
	int left = (rank + size - 1) % size;
	int right = (rank + size + 1) % size;
	int l_sendidx = rank;
	int l_recvidx = right;
	int r_sendidx = rank;
	int r_recvidx = left;
	int l_count = count / 2;
	int r_count = count - l_count;

	memcpy(&recvbuf[count * rank], sendbuf, sizeof(T) * count);
	for(int i = 1; i < size; ++i, ++l_sendidx, ++l_recvidx, --r_sendidx, --r_recvidx) {
		if(l_sendidx >= size) l_sendidx -= size;
		if(l_recvidx >= size) l_recvidx -= size;
		if(r_sendidx < 0) r_sendidx += size;
		if(r_recvidx < 0) r_recvidx += size;

		MPI_Request req[4];
		MPI_Irecv(&recvbuf[count * l_recvidx], l_count, MpiTypeOf<T>::type, right, PRM::MY_EXPAND_TAG, comm, &req[2]);
		MPI_Irecv(&recvbuf[count * r_recvidx + l_count], r_count, MpiTypeOf<T>::type, left, PRM::MY_EXPAND_TAG, comm, &req[3]);
		MPI_Isend(&recvbuf[count * l_sendidx], l_count, MpiTypeOf<T>::type, left, PRM::MY_EXPAND_TAG, comm, &req[0]);
		MPI_Isend(&recvbuf[count * r_sendidx + l_count], r_count, MpiTypeOf<T>::type, right, PRM::MY_EXPAND_TAG, comm, &req[1]);
		MPI_Waitall(4, req, MPI_STATUS_IGNORE);
	}
}

template <typename T>
void my_allgatherv(T *sendbuf, int send_count, T *recvbuf, int* recv_count, int* recv_offset, MPI_Comm comm)
{
	int size; MPI_Comm_size(comm, &size);
	int rank; MPI_Comm_rank(comm, &rank);
	int left = (rank + size - 1) % size;
	int right = (rank + size + 1) % size;
	int l_sendidx = rank;
	int l_recvidx = right;
	int r_sendidx = rank;
	int r_recvidx = left;

	memcpy(&recvbuf[recv_offset[rank]], sendbuf, sizeof(T) * send_count);
	for(int i = 1; i < size; ++i, ++l_sendidx, ++l_recvidx, --r_sendidx, --r_recvidx) {
		if(l_sendidx >= size) l_sendidx -= size;
		if(l_recvidx >= size) l_recvidx -= size;
		if(r_sendidx < 0) r_sendidx += size;
		if(r_recvidx < 0) r_recvidx += size;

		int l_send_off = recv_offset[l_sendidx];
		int l_send_cnt = recv_count[l_sendidx] / 2;
		int l_recv_off = recv_offset[l_recvidx];
		int l_recv_cnt = recv_count[l_recvidx] / 2;

		int r_send_off = recv_offset[r_sendidx] + recv_count[r_sendidx] / 2;
		int r_send_cnt = recv_count[r_sendidx] - recv_count[r_sendidx] / 2;
		int r_recv_off = recv_offset[r_recvidx] + recv_count[r_recvidx] / 2;
		int r_recv_cnt = recv_count[r_recvidx] - recv_count[r_recvidx] / 2;

		MPI_Request req[4];
		MPI_Irecv(&recvbuf[l_recv_off], l_recv_cnt, MpiTypeOf<T>::type, right, PRM::MY_EXPAND_TAG, comm, &req[2]);
		MPI_Irecv(&recvbuf[r_recv_off], r_recv_cnt, MpiTypeOf<T>::type, left, PRM::MY_EXPAND_TAG, comm, &req[3]);
		MPI_Isend(&recvbuf[l_send_off], l_send_cnt, MpiTypeOf<T>::type, left, PRM::MY_EXPAND_TAG, comm, &req[0]);
		MPI_Isend(&recvbuf[r_send_off], r_send_cnt, MpiTypeOf<T>::type, right, PRM::MY_EXPAND_TAG, comm, &req[1]);
		MPI_Waitall(4, req, MPI_STATUS_IGNORE);
	}
}
#endif

template <typename T>
void my_allgatherv(T *sendbuf, int send_count, T *recvbuf, int* recv_count, int* recv_offset, COMM_2D comm)
{
	memcpy(&recvbuf[recv_offset[comm.rank]], sendbuf, sizeof(T) * send_count);
	if(mpi.isMultiDimAvailable == false) {
		int size; MPI_Comm_size(comm.comm, &size);
		int rank; MPI_Comm_rank(comm.comm, &rank);
		int left = (rank + size - 1) % size;
		int right = (rank + size + 1) % size;
		my_allgatherv(recvbuf, recv_count, recv_offset, comm.comm, rank, size, left, right);
		return ;
	}
	{
		int size = comm.size_x;
		int rank = comm.rank % comm.size_x;
		int base = comm.rank - rank;
		int left = (rank + size - 1) % size + base;
		int right = (rank + size + 1) % size + base;
		my_allgatherv(recvbuf, &recv_count[base], &recv_offset[base], comm.comm, rank, size, left, right);
	}
	{
		int size = comm.size_y;
		int rank = comm.rank / comm.size_x;
		int left = compute_rank_2d(comm.rank_x, comm.rank_y - 1, comm.size_x, comm.size_y);
		int right = compute_rank_2d(comm.rank_x, comm.rank_y + 1, comm.size_x, comm.size_y);
		int count[comm.size_y];
		int offset[comm.size_y];
		for(int y = 0; y < comm.size_y; ++y) {
			int start = y * comm.size_x;
			int last = start + comm.size_x - 1;
			offset[y] = recv_offset[start];
			count[y] = recv_offset[last] + recv_count[last] - offset[y];
		}
		my_allgatherv(recvbuf, count, offset, comm.comm, rank, size, left, right);
	}
}

template <typename T>
void my_allgather(T *sendbuf, int count, T *recvbuf, COMM_2D comm)
{
	memcpy(&recvbuf[count * comm.rank], sendbuf, sizeof(T) * count);
	int recv_count[comm.size];
	int recv_offset[comm.size+1];
	recv_offset[0] = 0;
	for(int i = 0; i < comm.size; ++i) {
		recv_count[i] = count;
		recv_offset[i+1] = recv_offset[i] + count;
	}
	my_allgatherv(sendbuf, count, recvbuf, recv_count, recv_offset, comm);
}

} // namespace MpiCol {

//-------------------------------------------------------------//
// Multithread Partitioning and Scatter
//-------------------------------------------------------------//

// Usage: get_counts -> sum -> get_offsets
template <typename T>
class ParallelPartitioning
{
public:
	ParallelPartitioning(int num_partitions)
		: num_partitions_(num_partitions)
		, max_threads_(omp_get_max_threads())
		, thread_counts_(NULL)
		, thread_offsets_(NULL)
	{
		buffer_width_ = std::max<int>(CACHE_LINE/sizeof(T), num_partitions_);
		thread_counts_ = static_cast<T*>(cache_aligned_xmalloc(buffer_width_ * (max_threads_*2 + 1) * sizeof(T)));
		thread_offsets_ = thread_counts_ + buffer_width_*max_threads_;

		partition_size_ = static_cast<T*>(cache_aligned_xmalloc((num_partitions_*2 + 1) * sizeof(T)));
		partition_offsets_ = partition_size_ + num_partitions_;
	}
	~ParallelPartitioning()
	{
		free(thread_counts_);
		free(partition_size_);
	}
	T sum(T* base_offset = NULL) {
		const int width = buffer_width_;
		// compute sum of thread local count values
#pragma omp parallel for
		for(int r = 0; r < num_partitions_; ++r) {
			int sum = 0;
			for(int t = 0; t < max_threads_; ++t) {
				sum += thread_counts_[t*width + r];
			}
			partition_size_[r] = sum;
		}
		// compute offsets
		if(base_offset != NULL) {
#pragma omp parallel for
			for(int r = 0; r < num_partitions_; ++r) {
				partition_offsets_[r] = base_offset[r];
				base_offset[r] += partition_size_[r];
			}
#pragma omp parallel for
			for(int r = 0; r < num_partitions_; ++r) {
				thread_offsets_[0*width + r] = partition_offsets_[r];
				for(int t = 0; t < max_threads_; ++t) {
					thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
				}
			}
			return T(0);
		}
		else {
			partition_offsets_[0] = 0;
			for(int r = 0; r < num_partitions_; ++r) {
				partition_offsets_[r + 1] = partition_offsets_[r] + partition_size_[r];
			}
			// assert (send_counts[size] == bufsize*2);
			// compute offset of each threads
	#pragma omp parallel for
			for(int r = 0; r < num_partitions_; ++r) {
				thread_offsets_[0*width + r] = partition_offsets_[r];
				for(int t = 0; t < max_threads_; ++t) {
					thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
				}
				assert (thread_offsets_[max_threads_*width + r] == partition_offsets_[r + 1]);
			}
			return partition_offsets_[num_partitions_];
		}
	}
	T* get_counts() {
		T* counts = &thread_counts_[buffer_width_*omp_get_thread_num()];
		memset(counts, 0x00, buffer_width_*sizeof(T));
		return counts;
	}
	T* get_offsets() { return &thread_offsets_[buffer_width_*omp_get_thread_num()]; }

	const T* get_partition_offsets() const { return partition_offsets_; }
	const T* get_partition_size() const { return partition_size_; }

	bool check() const {
#ifndef	NDEBUG
		const int width = buffer_width_;
		// check offset of each threads
		for(int r = 0; r < num_partitions_; ++r) {
			assert (thread_offsets_[0*width + r] == partition_offsets_[r] + thread_counts_[0*width + r]);
			for(int t = 1; t < max_threads_; ++t) {
				assert (thread_offsets_[t*width + r] == thread_offsets_[(t-1)*width + r] + thread_counts_[t*width + r]);
			}
		}
#endif
		return true;
	}
private:
	int num_partitions_;
	int buffer_width_;
	int max_threads_;
	T* thread_counts_;
	T* thread_offsets_;
	T* partition_size_;
	T* partition_offsets_;
};

// Usage: get_counts -> sum -> get_offsets -> scatter -> gather
class ScatterContext
{
public:
	explicit ScatterContext(MPI_Comm comm)
		: comm_(comm)
		, max_threads_(omp_get_max_threads())
		, thread_counts_(NULL)
		, thread_offsets_(NULL)
		, send_counts_(NULL)
		, send_offsets_(NULL)
		, recv_counts_(NULL)
		, recv_offsets_(NULL)
	{
		MPI_Comm_size(comm_, &comm_size_);

		buffer_width_ = std::max<int>(CACHE_LINE/sizeof(int), comm_size_);
		thread_counts_ = static_cast<int*>(cache_aligned_xmalloc(buffer_width_ * (max_threads_*2 + 1) * sizeof(int)));
		thread_offsets_ = thread_counts_ + buffer_width_*max_threads_;

		send_counts_ = static_cast<int*>(cache_aligned_xmalloc((comm_size_*2 + 1) * 2 * sizeof(int)));
		send_offsets_ = send_counts_ + comm_size_;
		recv_counts_ = send_offsets_ + comm_size_ + 1;
		recv_offsets_ = recv_counts_ + comm_size_;
	}

	~ScatterContext()
	{
		::free(thread_counts_);
		::free(send_counts_);
	}

	int* get_counts() {
		int* counts = &thread_counts_[buffer_width_*omp_get_thread_num()];
		memset(counts, 0x00, buffer_width_*sizeof(int));
		return counts;
	}
	int* get_offsets() { return &thread_offsets_[buffer_width_*omp_get_thread_num()]; }

	void sum() {
		const int width = buffer_width_;
		// compute sum of thread local count values
#pragma omp parallel for if(comm_size_ > 1000)
		for(int r = 0; r < comm_size_; ++r) {
			int sum = 0;
			for(int t = 0; t < max_threads_; ++t) {
				sum += thread_counts_[t*width + r];
			}
			send_counts_[r] = sum;
		}
		// compute offsets
		send_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			send_offsets_[r + 1] = send_offsets_[r] + send_counts_[r];
		}
		// assert (send_counts[size] == bufsize*2);
		// compute offset of each threads
#pragma omp parallel for if(comm_size_ > 1000)
		for(int r = 0; r < comm_size_; ++r) {
			thread_offsets_[0*width + r] = send_offsets_[r];
			for(int t = 0; t < max_threads_; ++t) {
				thread_offsets_[(t+1)*width + r] = thread_offsets_[t*width + r] + thread_counts_[t*width + r];
			}
			assert (thread_offsets_[max_threads_*width + r] == send_offsets_[r + 1]);
		}
	}

	int get_send_count() { return send_offsets_[comm_size_]; }
	int get_recv_count() { return recv_offsets_[comm_size_]; }
	int* get_recv_offsets() { return recv_offsets_; }

	template <typename T>
	T* scatter(T* send_data) {
#ifndef	NDEBUG
		const int width = buffer_width_;
		// check offset of each threads
		for(int r = 0; r < comm_size_; ++r) {
			assert (thread_offsets_[0*width + r] == send_offsets_[r] + thread_counts_[0*width + r]);
			for(int t = 1; t < max_threads_; ++t) {
				assert (thread_offsets_[t*width + r] == thread_offsets_[(t-1)*width + r] + thread_counts_[t*width + r]);
			}
		}
#endif
		return MpiCol::alltoallv(send_data, send_counts_, send_offsets_,
				recv_counts_, recv_offsets_, comm_, comm_size_);
	}

	template <typename T>
	T* gather(T* send_data) {
		T* recv_data = static_cast<T*>(xMPI_Alloc_mem(send_offsets_[comm_size_] * sizeof(T)));
		MPI_Alltoallv(send_data, recv_counts_, recv_offsets_, MpiTypeOf<T>::type,
				recv_data, send_counts_, send_offsets_, MpiTypeOf<T>::type, comm_);
		return recv_data;
	}

	template <typename T>
	void free(T* buffer) {
		MPI_Free_mem(buffer);
	}

	void alltoallv(void* sendbuf, void* recvbuf, MPI_Datatype type, int recvbufsize)
	{
		recv_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			recv_offsets_[r + 1] = recv_offsets_[r] + recv_counts_[r];
		}
		MPI_Alltoall(send_counts_, 1, MPI_INT, recv_counts_, 1, MPI_INT, comm_);
		// calculate offsets
		recv_offsets_[0] = 0;
		for(int r = 0; r < comm_size_; ++r) {
			recv_offsets_[r + 1] = recv_offsets_[r] + recv_counts_[r];
		}
		if(recv_counts_[comm_size_] > recvbufsize) {
			fprintf(IMD_OUT, "Error: recv_counts_[comm_size_] > recvbufsize");
			throw "Error: buffer size not enough";
		}
		MPI_Alltoallv(sendbuf, send_counts_, send_offsets_, type,
				recvbuf, recv_counts_, recv_offsets_, type, comm_);
	}

private:
	MPI_Comm comm_;
	int comm_size_;
	int buffer_width_;
	int max_threads_;
	int* thread_counts_;
	int* thread_offsets_;
	int* restrict send_counts_;
	int* restrict send_offsets_;
	int* restrict recv_counts_;
	int* restrict recv_offsets_;

};

//-------------------------------------------------------------//
// MPI helper
//-------------------------------------------------------------//

namespace MpiCol {

template <typename Mapping>
void scatter(const Mapping mapping, int data_count, MPI_Comm comm)
{
	ScatterContext scatter(comm);
	typename Mapping::send_type* restrict partitioned_data = static_cast<typename Mapping::send_type*>(
						cache_aligned_xmalloc(data_count*sizeof(typename Mapping::send_type)));
#pragma omp parallel
	{
		int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			(counts[mapping.target(i)])++;
		} // #pragma omp for schedule(static)
	}

	scatter.sum();

#pragma omp parallel
	{
		int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			partitioned_data[(offsets[mapping.target(i)])++] = mapping.get(i);
		} // #pragma omp for schedule(static)
	} // #pragma omp parallel

	typename Mapping::send_type* recv_data = scatter.scatter(partitioned_data);
	int recv_count = scatter.get_recv_count();
	::free(partitioned_data); partitioned_data = NULL;

	int i;
#pragma omp parallel for lastprivate(i) schedule(static)
	for(i = 0; i < (recv_count&(~3)); i += 4) {
		mapping.set(i+0, recv_data[i+0]);
		mapping.set(i+1, recv_data[i+1]);
		mapping.set(i+2, recv_data[i+2]);
		mapping.set(i+3, recv_data[i+3]);
	} // #pragma omp parallel for
	for( ; i < recv_count; ++i) {
		mapping.set(i, recv_data[i]);
	}

	scatter.free(recv_data);
}

template <typename Mapping>
void gather(const Mapping mapping, int data_count, MPI_Comm comm)
{
	ScatterContext scatter(comm);

	int* restrict local_indices = static_cast<int*>(
			cache_aligned_xmalloc(data_count*sizeof(int)));
	typename Mapping::send_type* restrict partitioned_data = static_cast<typename Mapping::send_type*>(
			cache_aligned_xmalloc(data_count*sizeof(typename Mapping::send_type)));

#pragma omp parallel
	{
		int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			(counts[mapping.target(i)])++;
		} // #pragma omp for schedule(static)
	}

	scatter.sum();

#pragma omp parallel
	{
		int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
		for (int i = 0; i < data_count; ++i) {
			int pos = (offsets[mapping.target(i)])++;
			assert (pos < data_count);
			local_indices[i] = pos;
			partitioned_data[pos] = mapping.get(i);
			//// user defined ////
		} // #pragma omp for schedule(static)
	} // #pragma omp parallel

	// send and receive requests
	typename Mapping::send_type* restrict reply_verts = scatter.scatter(partitioned_data);
	int recv_count = scatter.get_recv_count();
	::free(partitioned_data);

	// make reply data
	typename Mapping::recv_type* restrict reply_data = static_cast<typename Mapping::recv_type*>(
			cache_aligned_xmalloc(recv_count*sizeof(typename Mapping::recv_type)));
#pragma omp parallel for
	for (int i = 0; i < recv_count; ++i) {
		reply_data[i] = mapping.map(reply_verts[i]);
	}
	scatter.free(reply_verts);

	// send and receive reply
	typename Mapping::recv_type* restrict recv_data = scatter.gather(reply_data);
	::free(reply_data);

	// apply received data to edges
#pragma omp parallel for
	for (int i = 0; i < data_count; ++i) {
		mapping.set(i, recv_data[local_indices[i]]);
	}

	scatter.free(recv_data);
	::free(local_indices);
}

} // namespace MpiCollective { //

//-------------------------------------------------------------//
// For print functions
//-------------------------------------------------------------//

double to_giga(int64_t v) { return v / (1024.0*1024.0*1024.0); }
double to_mega(int64_t v) { return v / (1024.0*1024.0); }
double diff_percent(int64_t v, int64_t sum, int demon) {
	double avg = sum / (double)demon;
	return (v - avg) / avg * 100.0;
}
double diff_percent(double v, double sum, int demon) {
	double avg = sum / (double)demon;
	return (v - avg) / avg * 100.0;
}
const char* minimum_type(int64_t max_value) {
	if(     max_value <= (int64_t(1) <<  7)) return "int8_t";
	else if(max_value <= (int64_t(1) <<  8)) return "uint8_t";
	else if(max_value <= (int64_t(1) << 15)) return "int16_t";
	else if(max_value <= (int64_t(1) << 16)) return "uint16_t";
	else if(max_value <= (int64_t(1) << 31)) return "int32_t";
	else if(max_value <= (int64_t(1) << 32)) return "uint32_t";
	else return "int64_t";
}

template <typename T> struct TypeName { };
template <> struct TypeName<int8_t> { static const char* value; };
const char* TypeName<int8_t>::value = "int8_t";
template <> struct TypeName<uint8_t> { static const char* value; };
const char* TypeName<uint8_t>::value = "uint8_t";
template <> struct TypeName<int16_t> { static const char* value; };
const char* TypeName<int16_t>::value = "int16_t";
template <> struct TypeName<uint16_t> { static const char* value; };
const char* TypeName<uint16_t>::value = "uint16_t";
template <> struct TypeName<int32_t> { static const char* value; };
const char* TypeName<int32_t>::value = "int32_t";
template <> struct TypeName<uint32_t> { static const char* value; };
const char* TypeName<uint32_t>::value = "uint32_t";
template <> struct TypeName<int64_t> { static const char* value; };
const char* TypeName<int64_t>::value = "int64_t";
template <> struct TypeName<uint64_t> { static const char* value; };
const char* TypeName<uint64_t>::value = "uint64_t";

//-------------------------------------------------------------//
// VarInt Encoding
//-------------------------------------------------------------//

namespace vlq {

enum CODING_ENUM {
	MAX_CODE_LENGTH_32 = 5,
	MAX_CODE_LENGTH_64 = 9,
};

#define VARINT_ENCODE_MACRO_32(p, v, l) \
if(v < 128) { \
	p[0] = (uint8_t)v; \
	l = 1; \
} \
else if(v < 128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7); \
	l = 2; \
} \
else if(v < 128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14); \
	l = 3; \
} \
else if(v < 128*128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21); \
	l = 4; \
} \
else { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28); \
	l = 5; \
}

#define VARINT_ENCODE_MACRO_64(p, v, l) \
if(v < 128) { \
	p[0] = (uint8_t)v; \
	l = 1; \
} \
else if(v < 128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7); \
	l = 2; \
} \
else if(v < 128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14); \
	l = 3; \
} \
else if(v < 128*128*128*128) { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21); \
	l = 4; \
} \
else if(v < 128LL*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28); \
	l = 5; \
} \
else if(v < 128LL*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35); \
	l = 6; \
} \
else if(v < 128LL*128*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42); \
	l = 7; \
} \
else if(v < 128LL*128*128*128*128*128*128*128){ \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42) | 0x80; \
	p[7]= (uint8_t)(v >> 49); \
	l = 8; \
} \
else { \
	p[0]= (uint8_t)v | 0x80; \
	p[1]= (uint8_t)(v >> 7) | 0x80; \
	p[2]= (uint8_t)(v >> 14) | 0x80; \
	p[3]= (uint8_t)(v >> 21) | 0x80; \
	p[4]= (uint8_t)(v >> 28) | 0x80; \
	p[5]= (uint8_t)(v >> 35) | 0x80; \
	p[6]= (uint8_t)(v >> 42) | 0x80; \
	p[6]= (uint8_t)(v >> 49) | 0x80; \
	p[8]= (uint8_t)(v >> 56); \
	l = 9; \
}

#define VARINT_DECODE_MACRO_32(p, v, l) \
if(p[0] < 128) { \
	v = p[0]; \
	l = 1; \
} \
else if(p[1] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)p[1] << 7); \
	l = 2; \
} \
else if(p[2] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2]) << 14); \
	l = 3; \
} \
else if(p[3] < 128) { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2] & 0x7F) << 14) | ((uint32_t)(p[3]) << 21); \
	l = 4; \
} \
else { \
	v = (p[0] & 0x7F) | ((uint32_t)(p[1] & 0x7F) << 7) | \
			((uint32_t)(p[2] & 0x7F) << 14) | ((uint32_t)(p[3] & 0x7F) << 21) | \
			((uint32_t)(p[4]) << 28); \
	l = 5; \
}

#define VARINT_DECODE_MACRO_64(p, v, l) \
if(p[0] < 128) { \
	v = p[0]; \
	l = 1; \
} \
else if(p[1] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)p[1] << 7); \
	l = 2; \
} \
else if(p[2] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2]) << 14); \
	l = 3; \
} \
else if(p[3] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3]) << 21); \
	l = 4; \
} \
else if(p[4] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4]) << 28); \
	l = 5; \
} \
else if(p[5] < 128) { \
	v= (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5]) << 35); \
	l = 6; \
} \
else if(p[6] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6]) << 42); \
	l = 7; \
} \
else if(p[7] < 128) { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6] & 0x7F) << 42) | ((uint64_t)(p[7]) << 49); \
	l = 8; \
} \
else { \
	v = (p[0] & 0x7F) | ((uint64_t)(p[1] & 0x7F) << 7) | \
			((uint64_t)(p[2] & 0x7F) << 14) | ((uint64_t)(p[3] & 0x7F) << 21) | \
			((uint64_t)(p[4] & 0x7F) << 28) | ((uint64_t)(p[5] & 0x7F) << 35) | \
			((uint64_t)(p[6] & 0x7F) << 42) | ((uint64_t)(p[7] & 0x7F) << 49) | \
			((uint64_t)(p[8]) << 56); \
	l = 9; \
}

int encode(const uint32_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint32_t v = input[k];
		int len;
		VARINT_ENCODE_MACRO_32(p, v, len);
		p += len;
	}
	return p - output;
}

int encode(const uint64_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint64_t v = input[k];
		int len;
		VARINT_ENCODE_MACRO_64(p, v, len);
		p += len;
	}
	return p - output;
}

int encode_signed(const uint64_t* input, int length, uint8_t* output)
{
	uint8_t* p = output;
	for(int k = 0; k < length; ++k) {
		uint64_t v = input[k];
		v = (v << 1) ^ (((int64_t)v) >> 63);
		int len;
		VARINT_ENCODE_MACRO_64(p, v, len);
		p += len;
	}
	return p - output;
}

int decode(const uint8_t* input, int length, uint32_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint32_t v;
		int len;
		VARINT_DECODE_MACRO_32(p, v, len);
		output[n] = v;
		p += len;
	}
	return n;
}

int decode(const uint8_t* input, int length, uint64_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint64_t v;
		int len;
		VARINT_DECODE_MACRO_64(p, v, len);
		output[n] = v;
		p += len;
	}
	return n;
}

int decode_signed(const uint8_t* input, int length, uint64_t* output)
{
	const uint8_t* p = input;
	const uint8_t* p_end = input + length;
	int n = 0;
	for(; p < p_end; ++n) {
		uint64_t v;
		int len;
		VARINT_DECODE_MACRO_64(p, v, len);
		output[n] = (v >> 1) ^ (((int64_t)(v << 63)) >> 63);
		p += len;
	}
	return n;
}

int encode_gpu_compat(const uint32_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = MAX_CODE_LENGTH_32, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			uint32_t v = input[i + k];
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_32(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int encode_gpu_compat(const uint64_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = MAX_CODE_LENGTH_64, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			uint64_t v = input[i + k];
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_64(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int encode_gpu_compat_signed(const int64_t* input, int length, uint8_t* output)
{
	enum { MAX_CODE_LENGTH = MAX_CODE_LENGTH_64, SIMD_WIDTH = 32 };
	uint8_t tmp_buffer[SIMD_WIDTH][MAX_CODE_LENGTH];
	uint8_t code_length[SIMD_WIDTH];
	int count[MAX_CODE_LENGTH + 1];

	uint8_t* out_ptr = output;
	for(int i = 0; i < length; i += SIMD_WIDTH) {
		int width = std::min(length - i, (int)SIMD_WIDTH);

		for(int k = 0; k < MAX_CODE_LENGTH; ++k) {
			count[k] = 0;
		}
		count[MAX_CODE_LENGTH] = 0;

		for(int k = 0; k < width; ++k) {
			int64_t v_raw = input[i + k];
			uint64_t v = (v_raw < 0) ? ((uint64_t)(~v_raw) << 1) | 1 : ((uint64_t)v_raw << 1);
		//	uint64_t v = (v_raw << 1) ^ (v_raw >> 63);
			assert ((int64_t)v >= 0);
			uint8_t* dst = tmp_buffer[k];
			int len;
			VARINT_ENCODE_MACRO_64(dst, v, len);
			code_length[k] = len;
			for(int r = 0; r < len; ++r) {
				++count[r + 1];
			}
		}

		for(int k = 1; k < MAX_CODE_LENGTH; ++k) count[k + 1] += count[k];

		for(int k = 0; k < width; ++k) {
			for(int r = 0; r < code_length[k]; ++r) {
				out_ptr[count[r]++] = tmp_buffer[k][r];
			}
		}

		out_ptr += count[MAX_CODE_LENGTH];
	}

	return out_ptr - output;
}

int sparsity_factor(int64_t range, int64_t num_values)
{
	if(num_values == 0) return 0;
	const double sparsity = (double)range / (double)num_values;
	int scale;
	if(sparsity < 1.0)
		scale = 1;
	else if(sparsity < 128)
		scale = 2;
	else if(sparsity < 128LL*128)
		scale = 3;
	else if(sparsity < 128LL*128*128)
		scale = 4;
	else if(sparsity < 128LL*128*128*128)
		scale = 5;
	else if(sparsity < 128LL*128*128*128*128)
		scale = 6;
	else if(sparsity < 128LL*128*128*128*128*128)
		scale = 7;
	else if(sparsity < 128LL*128*128*128*128*128*128)
		scale = 8;
	else if(sparsity < 128LL*128*128*128*128*128*128*128)
		scale = 9;
	else
		scale = 10;
	return scale;
}

struct PacketIndex {
	uint32_t offset;
	uint16_t length;
	uint16_t num_int;
};

class BitmapEncoder {
public:

	static int calc_max_packet_size(int64_t max_data_size) {
		int max_threads = omp_get_max_threads();
		return ((max_data_size/max_threads) > 32*1024) ? 16*1024 : 256;
	}

	static int64_t calc_capacity_of_values(int64_t bitmap_size, int num_bits_per_word, int64_t max_data_size) {
		int64_t num_bits = bitmap_size * num_bits_per_word;
		int packet_overhead = sizeof(PacketIndex) + MAX_CODE_LENGTH_64;

		int max_packet_size = calc_max_packet_size(max_data_size);
		int packet_min_bytes = max_packet_size - MAX_CODE_LENGTH_64*2;

		int64_t min_data_bytes = num_bits / 8 - max_packet_size * omp_get_max_threads();
		double overhead_factor = 1 + (double)packet_overhead / (double)packet_min_bytes;
		return int64_t(min_data_bytes / overhead_factor) - (num_bits / 128);
	}

	/**
	 * bitmap is executed along only one pass
	 * BitmapF::operator (int64_t)
	 * BitmapF::BitsPerWord
	 * BitmapF::BitmapType
	 */
	template <typename BitmapF, bool b64>
	bool bitmap_to_stream(
			const BitmapF& bitmap, int64_t bitmap_size,
			void* output, int64_t* data_size,
			int64_t max_size)
	{
		typedef typename BitmapF::BitmapType BitmapType;

		out_len = max_size;
		head = sizeof(uint32_t);
		tail = max_size - sizeof(PacketIndex);
		outbuf = output;

		assert ((max_size % sizeof(uint32_t)) == 0);
		const int max_threads = omp_get_max_threads();
		const int max_packet_size = calc_max_packet_size(max_size);
		const int64_t threshold = max_size;
		bool b_break = false;

#pragma omp parallel reduction(|:b_break)
		{
			uint8_t* buf;
			PacketIndex* pk_idx;
			int remain_packet_length = max_packet_size;
			if(reserve_packet(&buf, &pk_idx, max_packet_size) == false) {
				throw_exception("Not enough buffer: bitmap_to_stream");
			}
			uint8_t* ptr = buf;
			int num_int = 0;

			int64_t chunk_size = (bitmap_size + max_threads - 1) / max_threads;
			int64_t i_start = chunk_size * omp_get_thread_num();
			int64_t i_end = std::min(i_start + chunk_size, bitmap_size);
			int64_t prev_val = 0;

			for(int64_t i = i_start; i < i_end; ++i) {
				BitmapType bmp_val = bitmap(i);
				while(bmp_val != BitmapType(0)) {
					uint32_t bit_idx = __builtin_ctzl(bmp_val);
					int64_t new_val = BitmapF::BitsPerWord * i + bit_idx;
					int64_t diff = new_val - prev_val;

					if(remain_packet_length < (b64 ? MAX_CODE_LENGTH_64 : MAX_CODE_LENGTH_32)) {
						pk_idx->length = ptr - buf;
						pk_idx->num_int = num_int;
						if(reserve_packet(&buf, &pk_idx, max_packet_size) == false) {
							b_break = true;
							i = i_end;
							break;
						}
						num_int = 0;
						remain_packet_length = max_packet_size;
					}

					int len;
					if(b64) { VARINT_ENCODE_MACRO_64(ptr, diff, len); }
					else { VARINT_ENCODE_MACRO_32(ptr, diff, len); }
					ptr += len;
					++num_int;

					prev_val = new_val;
					bmp_val &= bmp_val - 1;
				}
			}
		} // #pragma omp parallel reduction(|:b_break)

		if(b_break) {
			*data_size = threshold;
			return false;
		}

		*data_size = compact_output();
		return true;
	}
private:
	int64_t head, tail;
	int64_t out_len;
	uint8_t* outbuf;

	bool reserve_packet(uint8_t** ptr, PacketIndex** pk_idx, int req_size) {
		assert ((req_size % sizeof(uint32_t)) == 0);
		int64_t next_head, next_tail;
#pragma omp critical
		{
			*ptr = outbuf + head;
			next_head = head = head + req_size;
			next_tail = tail = tail - sizeof(PacketIndex);
		}
		*pk_idx = (PacketIndex*)&outbuf[next_tail];
		(*pk_idx)->offset = head / sizeof(uint32_t);
		(*pk_idx)->length = 0;
		(*pk_idx)->num_int = 0;
		return next_head <= next_tail;
	}

	int64_t compact_output() {
		int num_packet = (out_len - tail) / sizeof(PacketIndex) - 1;
		PacketIndex* pk_tail = (PacketIndex*)&outbuf[out_len] - 2;
		PacketIndex* pk_head = (PacketIndex*)&outbuf[out_len - tail];
		for(int i = 0; i < (num_packet/2); ++i) {
			std::swap(pk_tail[-i], pk_head[i]);
		}
		pk_tail[1].offset = tail / sizeof(uint32_t); // bann hei

#define O_TO_S(offset) ((offset)*sizeof(uint32_t))
#define L_TO_S(length) roundup<int>(length, sizeof(uint32_t))
#define TO_S(offset, length) (O_TO_S(offset) + L_TO_S(length))
		int i = 0;
		for( ; i < num_packet; ++i) {
			// When the empty region length is larger than 32 bytes, break.
			if(O_TO_S(pk_head[i+1].offset - pk_head[i].offset) - L_TO_S(pk_head[i].length) > 32)
				break;
		}
		VERVOSE(print_with_prefix("Move %ld length", out_len - sizeof(PacketIndex) - O_TO_S(pk_head[i+1].offset)));
		for( ; i < num_packet; ++i) {
			memmove(outbuf + TO_S(pk_head[i].offset, pk_head[i].length),
					outbuf + O_TO_S(pk_head[i+1].offset),
					(i+1 < num_packet) ? L_TO_S(pk_head[i+1].length) : num_packet*sizeof(PacketIndex));
			pk_head[i+1].offset = pk_head[i].offset + L_TO_S(pk_head[i].length) / sizeof(uint32_t);
		}

		*(uint32_t*)outbuf = pk_head[num_packet].offset;
		return O_TO_S(pk_head[num_packet].offset) + num_packet*sizeof(PacketIndex);
#undef O_TO_S
#undef L_TO_S
#undef TO_S
	}

}; // class BitmapEncoder

template <typename Callback, bool b64>
void decode_stream(void* stream, int64_t data_size, Callback cb) {
	assert (data_size >= 4);
	uint8_t* srcbuf = (uint8_t*)stream;
	uint32_t packet_index_start = *(uint32_t*)srcbuf;
	int64_t pk_offset = packet_index_start * sizeof(uint32_t);
	int num_packets = (data_size - pk_offset) / sizeof(PacketIndex);
	PacketIndex* pk_head = (PacketIndex*)(srcbuf + pk_offset);

	for(int i = 0; i < num_packets; ++i) {
		uint8_t* ptr = srcbuf + pk_head[i].offset * sizeof(uint32_t);
		int num_int = pk_head[i].num_int;
		for(int c = 0; c < num_int; ++c) {
			int len;
			if(b64) { int64_t v; VARINT_DECODE_MACRO_64(ptr, v, len); cb(v); }
			else { int32_t v; VARINT_DECODE_MACRO_32(ptr, v, len); cb(v); }
			ptr += len;
		}
		assert (ptr == srcbuf + pk_head[i].length + pk_head[i].offset * sizeof(uint32_t));
	}
}

} // namespace vlq {

namespace memory {

template <typename T>
class Pool {
public:
	Pool()
	{
	}
	virtual ~Pool() {
		clear_();
	}

	virtual T* get() {
		if(free_list_.empty()) {
			return allocate_new();
		}
		T* buffer = free_list_.back();
		free_list_.pop_back();
		return buffer;
	}

	virtual void free(T* buffer) {
		free_list_.push_back(buffer);
	}

	virtual void clear() {
		clear_();
	}

	bool empty() const {
		return free_list_.size() == 0;
	}

	size_t size() const {
		return free_list_.size();
	}

protected:
	std::vector<T*> free_list_;

	virtual T* allocate_new() {
		return new (malloc(sizeof(T))) T();
	}

private:
	void clear_() {
		for(int i = 0; i < (int)free_list_.size(); ++i) {
			free_list_[i]->~T();
			::free(free_list_[i]);
		}
		free_list_.clear();
	}
};

//! Only get() and free() are thread-safe. The other functions are NOT thread-safe.
template <typename T>
class ConcurrentPool : public Pool<T> {
	typedef Pool<T> super_;
public:
	ConcurrentPool()
		: Pool<T>()
	{
		pthread_mutex_init(&thread_sync_, NULL);
	}
	virtual ~ConcurrentPool()
	{
		pthread_mutex_lock(&thread_sync_);
	}

	virtual T* get() {
		pthread_mutex_lock(&thread_sync_);
		if(this->free_list_.empty()) {
			pthread_mutex_unlock(&thread_sync_);
			T* new_buffer = this->allocate_new();
			return new_buffer;
		}
		T* buffer = this->free_list_.back();
		this->free_list_.pop_back();
		pthread_mutex_unlock(&thread_sync_);
		return buffer;
	}

	virtual void free(T* buffer) {
		pthread_mutex_lock(&thread_sync_);
		this->free_list_.push_back(buffer);
		pthread_mutex_unlock(&thread_sync_);
	}

	virtual void clear() {
		pthread_mutex_lock(&thread_sync_);
		super_::clear();
		pthread_mutex_unlock(&thread_sync_);
	}

	/*
	bool empty() const { return super_::empty(); }
	size_t size() const { return super_::size(); }
	void clear() { super_::clear(); }
	*/
protected:
	pthread_mutex_t thread_sync_;
};

template <typename T>
class vector_w : public std::vector<T*>
{
	typedef std::vector<T*> super_;
public:
	~vector_w() {
		for(typename super_::iterator it = this->begin(); it != this->end(); ++it) {
			(*it)->~T();
			::free(*it);
		}
		super_::clear();
	}
};

template <typename T>
class deque_w : public std::deque<T*>
{
	typedef std::deque<T*> super_;
public:
	~deque_w() {
		for(typename super_::iterator it = this->begin(); it != this->end(); ++it) {
			(*it)->~T();
			::free(*it);
		}
		super_::clear();
	}
};

template <typename T>
class Store {
public:
	Store() {
	}
	void init(Pool<T>* pool) {
		pool_ = pool;
		filled_length_ = 0;
		buffer_length_ = 0;
		resize_buffer(16);
	}
	~Store() {
		for(int i = 0; i < filled_length_; ++i){
			pool_->free(buffer_[i]);
		}
		filled_length_ = 0;
		buffer_length_ = 0;
		::free(buffer_); buffer_ = NULL;
	}

	void submit(T* value) {
		const int offset = filled_length_++;

		if(buffer_length_ == filled_length_)
			expand();

		buffer_[offset] = value;
	}

	void clear() {
		for(int i = 0; i < filled_length_; ++i){
			buffer_[i]->clear();
			assert (buffer_[i]->size() == 0);
			pool_->free(buffer_[i]);
		}
		filled_length_ = 0;
	}

	T* front() {
		if(filled_length_ == 0) {
			push();
		}
		return buffer_[filled_length_ - 1];
	}

	void push() {
		submit(pool_->get());
	}

	int64_t size() const { return filled_length_; }
	T* get(int index) const { return buffer_[index]; }
private:

	void resize_buffer(int allocation_size)
	{
		T** new_buffer = (T**)malloc(allocation_size*sizeof(buffer_[0]));
		if(buffer_length_ != 0) {
			memcpy(new_buffer, buffer_, filled_length_*sizeof(buffer_[0]));
			::free(buffer_);
		}
		buffer_ = new_buffer;
		buffer_length_ = allocation_size;
	}

	void expand()
	{
		if(filled_length_ == buffer_length_)
			resize_buffer(std::max<int64_t>(buffer_length_*2, 16));
	}

	int64_t filled_length_;
	int64_t buffer_length_;
	T** buffer_;
	Pool<T>* pool_;
};

template <typename T>
class ConcurrentStack
{
public:
	ConcurrentStack()
	{
		pthread_mutex_init(&thread_sync_, NULL);
	}

	~ConcurrentStack()
	{
		pthread_mutex_destroy(&thread_sync_);
	}

	void push(const T& d)
	{
		pthread_mutex_lock(&thread_sync_);
		stack_.push_back(d);
		pthread_mutex_unlock(&thread_sync_);
	}

	bool pop(T* ret)
	{
		pthread_mutex_lock(&thread_sync_);
		if(stack_.size() == 0) {
			pthread_mutex_unlock(&thread_sync_);
			return false;
		}
		*ret = stack_.back(); stack_.pop_back();
		pthread_mutex_unlock(&thread_sync_);
		return true;
	}

	std::vector<T> stack_;
	pthread_mutex_t thread_sync_;
};

struct SpinBarrier {
	volatile int step, cnt;
	int max;
	explicit SpinBarrier(int num_threads) {
		step = cnt = 0;
		max = num_threads;
	}
	void barrier() {
		int cur_step = step;
		int wait_cnt = __sync_add_and_fetch(&cnt, 1);
		assert (wait_cnt <= max);
		if(wait_cnt == max) {
			cnt = 0;
			__sync_add_and_fetch(&step, 1);
			return ;
		}
		while(step == cur_step) ;
	}
};

void copy_mt(void* dst, void* src, size_t size) {
#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int64_t i_start, i_end;
		get_partition<int64_t>(size, num_threads, tid, i_start, i_end);
		memcpy((int8_t*)dst + i_start, (int8_t*)src + i_start, i_end - i_start);
	}
}

} // namespace memory

namespace profiling {

class ProfilingInformationStore {
public:
	void submit(double span, const char* content, int number) {
#pragma omp critical (pis_submit_time)
		times_.push_back(TimeElement(span, content, number));
	}
	void submit(int64_t span_micro, const char* content, int number) {
#pragma omp critical (pis_submit_time)
		times_.push_back(TimeElement((double)span_micro / 1000000.0, content, number));
	}
	void submitCounter(int64_t counter, const char* content, int number) {
#pragma omp critical (pis_submit_counter)
		counters_.push_back(CountElement(counter, content, number));
	}
	void reset() {
		times_.clear();
		counters_.clear();
	}
	void printResult() {
		printTimeResult();
		printCountResult();
	}
private:
	struct TimeElement {
		double span;
		const char* content;
		int number;

		TimeElement(double span__, const char* content__, int number__)
			: span(span__), content(content__), number(number__) { }
	};
	struct CountElement {
		int64_t count;
		const char* content;
		int number;

		CountElement(int64_t count__, const char* content__, int number__)
			: count(count__), content(content__), number(number__) { }
	};

	void printTimeResult() {
		int num_times = times_.size();
		double *dbl_times = new double[num_times];
		double *sum_times = new double[num_times];
		double *max_times = new double[num_times];

		for(int i = 0; i < num_times; ++i) {
			dbl_times[i] = times_[i].span;
		}

		MPI_Reduce(dbl_times, sum_times, num_times, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(dbl_times, max_times, num_times, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		if(mpi.isMaster()) {
			for(int i = 0; i < num_times; ++i) {
				fprintf(IMD_OUT, "Time of %s, %d, Avg, %f, Max, %f, (ms)\n", times_[i].content,
						times_[i].number,
						sum_times[i] / mpi.size_2d * 1000.0,
						max_times[i] * 1000.0);
			}
		}

		delete [] dbl_times;
		delete [] sum_times;
		delete [] max_times;
	}

	double displayValue(int64_t value) {
		if(value < int64_t(1000))
			return (double)value;
		else if(value < int64_t(1000)*1000)
			return value / 1000.0;
		else if(value < int64_t(1000)*1000*1000)
			return value / (1000.0*1000);
		else if(value < int64_t(1000)*1000*1000*1000)
			return value / (1000.0*1000*1000);
		else
			return value / (1000.0*1000*1000*1000);
	}

	const char* displaySuffix(int64_t value) {
		if(value < int64_t(1000))
			return "";
		else if(value < int64_t(1000)*1000)
			return "K";
		else if(value < int64_t(1000)*1000*1000)
			return "M";
		else if(value < int64_t(1000)*1000*1000*1000)
			return "G";
		else
			return "T";
	}

	void printCountResult() {
		int num_times = counters_.size();
		int64_t *dbl_times = new int64_t[num_times];
		int64_t *sum_times = new int64_t[num_times];
		int64_t *max_times = new int64_t[num_times];

		for(int i = 0; i < num_times; ++i) {
			dbl_times[i] = counters_[i].count;
		}

		MPI_Reduce(dbl_times, sum_times, num_times, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(dbl_times, max_times, num_times, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);

		if(mpi.isMaster()) {
			for(int i = 0; i < num_times; ++i) {
				int64_t sum = sum_times[i], avg = sum_times[i] / mpi.size_2d, maximum = max_times[i];
				fprintf(IMD_OUT, "%s, %d, Sum, %ld, Avg, %ld, Max, %ld\n", counters_[i].content,
						counters_[i].number, sum, avg, maximum);
			}
		}

		delete [] dbl_times;
		delete [] sum_times;
		delete [] max_times;
	}

	std::vector<TimeElement> times_;
	std::vector<CountElement> counters_;
};

ProfilingInformationStore g_pis;

class TimeKeeper {
public:
	TimeKeeper() : start_(get_time_in_microsecond()){ }
	void submit(const char* content, int number) {
		int64_t end = get_time_in_microsecond();
		g_pis.submit(end - start_, content, number);
		start_ = end;
	}
	int64_t getSpanAndReset() {
		int64_t end = get_time_in_microsecond();
		int64_t span = end - start_;
		start_ = end;
		return span;
	}
private:
	int64_t start_;
};

class TimeSpan {
	TimeSpan(int64_t init) : span_(init) { }
public:
	TimeSpan() : span_(0) { }
	TimeSpan(TimeKeeper& keeper) : span_(keeper.getSpanAndReset()) { }

	void reset() { span_ = 0; }
	TimeSpan& operator += (TimeKeeper& keeper) {
		__sync_fetch_and_add(&span_, keeper.getSpanAndReset());
		return *this;
	}
	TimeSpan& operator -= (TimeKeeper& keeper) {
		__sync_fetch_and_add(&span_, - keeper.getSpanAndReset());
		return *this;
	}
	TimeSpan& operator += (TimeSpan span) {
		__sync_fetch_and_add(&span_, span.span_);
		return *this;
	}
	TimeSpan& operator -= (TimeSpan span) {
		__sync_fetch_and_add(&span_, - span.span_);
		return *this;
	}
	TimeSpan& operator += (int64_t span) {
		__sync_fetch_and_add(&span_, span);
		return *this;
	}
	TimeSpan& operator -= (int64_t span) {
		__sync_fetch_and_add(&span_, - span);
		return *this;
	}

	TimeSpan operator + (TimeSpan span) {
		return TimeSpan(span_ + span.span_);
	}
	TimeSpan operator - (TimeSpan span) {
		return TimeSpan(span_ - span.span_);
	}

	void submit(const char* content, int number) {
		g_pis.submit(span_, content, number);
		span_ = 0;
	}
	double getSpan() {
		return (double)span_ / 1000000.0;
	}
private:
	int64_t span_;
};

} // namespace profiling

#if BACKTRACE_ON_SIGNAL

namespace backtrace {

#define SPIN_LOCK(lock) pthread_mutex_lock(&(lock))
#define SPIN_UNLOCK(lock) pthread_mutex_unlock(&(lock))

struct StackFrame {
	PROF(int64_t enter_clock;)
	const char* name;
	int line;
};

struct PrintBuffer {
	char* buffer;
	int length;
	int capacity;

	PrintBuffer() {
		length = 0;
		capacity = 16*1024*1024;
		buffer = (char*)malloc(capacity);
		buffer[0] = '\0';
	}

	~PrintBuffer() {
		free(buffer); buffer = NULL;
	}

	void clear() {
		length = 0;
		buffer[0] = '\0';
	}

	void add(const char* fmt, va_list arg) {
	}
};

struct ThreadStack {
	int tid;
	pthread_mutex_t* lock;
	std::vector<StackFrame>* frames;
	PrintBuffer* pbuf;
};

std::vector<ThreadStack>* thread_stacks = NULL;
pthread_mutex_t thread_stack_lock = PTHREAD_MUTEX_INITIALIZER;
volatile int next_thread_id = 0;
volatile bool finish_backtrace_thread = false;
pthread_t backtrace_thread;

__thread bool disable_trace = false;
__thread int thread_id = -1;
__thread pthread_mutex_t stack_frame_lock = PTHREAD_MUTEX_INITIALIZER;
__thread std::vector<StackFrame>* stack_frames = NULL;
__thread PrintBuffer* pbuf;

void* backtrace_thread_routine(void* p) {
	char filename[300];
	int print_count = 0;
	sprintf(filename, "log-backtrace.%d", mpi.rank);
	FILE* fp = fopen(filename, "w");
	if(fp == NULL) {
		throw_exception("failed to open the file %s", filename);
	}
	fprintf(fp, "===== Backtrace File Rank=%d =====\n", mpi.rank);

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, PRINT_BT_SIGNAL);

	int sig;
	while(sigwait(&set, &sig) == 0) {
		if(finish_backtrace_thread) {
			fclose(fp); fp = NULL;
			return NULL;
		}
		fprintf(fp, "======= Print Backtrace (%d-th) =======\n", print_count++);

		// disable tracing to avoid modifying data during printing
		disable_trace = true;
		SPIN_LOCK(thread_stack_lock);
		if(thread_stacks != NULL) {
			for(int i = 0; i < int(thread_stacks->size()); ++i) {
				ThreadStack th = (*thread_stacks)[i];
				SPIN_LOCK(*th.lock);
				const std::vector<StackFrame>& frames = *(th.frames);
				int num_frames = int(frames.size());
				fprintf(fp, "Thread %d:\n", th.tid);
				for(int s = 0; s < num_frames; ++s) {
					const StackFrame& frame = frames[num_frames - s - 1];
					fprintf(fp, "    %2d:"PROF("[%f]")" %s:%d\n", s,
#if PROFILING_MODE
							(double)frame.enter_clock / 1000000.0,
#endif
							frame.name, frame.line);
				}
				SPIN_UNLOCK(*th.lock);
			}
		}
		SPIN_UNLOCK(thread_stack_lock);
		// restart tracing
		disable_trace = false;

		fprintf(fp, "============= Backtrace end ===========\n");
		fflush(fp);
	}

	throw_exception("Error on sigwait");
	return NULL;
}

void buffered_print(const char* format, ...) {
	char buf[300];
	SPIN_LOCK(stack_frame_lock);
	va_list arg;
	va_start(arg, format);
    vsnprintf(buf, sizeof(buf), format, arg);
    va_end(arg);
	if(pbuf->length + sizeof(buf) + 1 >= pbuf->capacity) {
		pbuf->capacity *= 2;
		pbuf->buffer = (char*)realloc(pbuf->buffer, pbuf->capacity);
	}
	pbuf->length += snprintf(pbuf->buffer + pbuf->length, sizeof(buf),
			"[r:%d,%f] %s\n", mpi.rank, global_clock.get() / 1000000.0, buf);
	SPIN_UNLOCK(stack_frame_lock);
}

void start_thread() {
	pthread_create(&backtrace_thread, NULL, backtrace_thread_routine, NULL);
}

void thread_join() {
	finish_backtrace_thread = true;
	pthread_kill(backtrace_thread, PRINT_BT_SIGNAL);
	pthread_join(backtrace_thread, NULL);
	SPIN_LOCK(thread_stack_lock);
	delete thread_stacks; thread_stacks = NULL;
	SPIN_UNLOCK(thread_stack_lock);
}

} // namespace backtrace {

extern "C" void user_defined_proc(const int *FLAG, const char *NAME, const int *LINE, const int *THREAD) {
	using namespace backtrace;

	if(disable_trace) {
		return ;
	}

	if(thread_id == -1) {
		// initialize thread local storage
		thread_id = __sync_fetch_and_add(&next_thread_id, 1);
		stack_frames = new std::vector<StackFrame>();
		pbuf = new PrintBuffer();
		// add thread info to thread_stacks
		ThreadStack th;
		th.tid = thread_id;
		th.lock = &stack_frame_lock;
		th.frames = stack_frames;
		th.pbuf = pbuf;

		SPIN_LOCK(thread_stack_lock);
		if(finish_backtrace_thread == false) {
			if(thread_stacks == NULL) {
				thread_stacks = new std::vector<ThreadStack>();
			}
			thread_stacks->push_back(th);
		}
		SPIN_UNLOCK(thread_stack_lock);
	}

	SPIN_LOCK(stack_frame_lock);
	if(finish_backtrace_thread) {
		if(stack_frames != NULL) {
			delete stack_frames; stack_frames = NULL;
			delete pbuf; pbuf = NULL;
		}
	}
	else {
		switch(*FLAG) {
		case 2:
		case 4:
		case 102:
		{
			StackFrame frame;
			PROF(frame.enter_clock = global_clock.get());
			frame.name = NAME;
			frame.line = *LINE;
			stack_frames->push_back(frame);
		}
			break;
		case 3:
		case 5:
		case 103:
		{
			assert(stack_frames->size() > 0);
			stack_frames->pop_back();
		}
			break;
		default:
			break;
		}
	}
	SPIN_UNLOCK(stack_frame_lock);
}

#undef SPIN_LOCK
#undef SPIN_UNLOCK

#endif // #if BACKTRACE_ON_SIGNAL

#if VERVOSE_MODE
volatile int64_t g_tp_comm;
volatile int64_t g_bu_pred_comm;
volatile int64_t g_bu_bitmap_comm;
volatile int64_t g_bu_list_comm;
volatile int64_t g_expand_bitmap_comm;
volatile int64_t g_expand_list_comm;
volatile double g_gpu_busy_time;
#endif

/* edgefactor = 16, seed1 = 2, seed2 = 3 */
int64_t pf_nedge[] = {
	-1,
	32, // 1
	64,
	128,
	256,
	512,
	1024,
	2048,
	4096 , // 8
	8192 ,
	16383 ,
	32767 ,
	65535 ,
	131070 ,
	262144 ,
	524285 ,
	1048570 ,
	2097137 ,
	4194250 ,
	8388513 ,
	16776976 ,
	33553998 ,
	67108130 ,
	134216177 ,
	268432547 ,
	536865258 ,
	1073731075 ,
	2147462776 ,
	4294927670 ,
	8589858508 ,
	17179724952 ,
	34359466407 ,
	68718955183 , // = 2^36 - 521553
	137437972330, // 33
	274876029861, // 34
	549752273512, // 35
	1099505021204, // 36
	0, // 37
	0, // 38
	0, // 39
	0, // 40
	0, // 41
	0 // 42
};

#endif /* UTILS_IMPL_HPP_ */
