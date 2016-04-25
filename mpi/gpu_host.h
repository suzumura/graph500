/*
 * gpu_host.h
 *
 *  Created on: Mar 27, 2012
 *      Author: koji
 */

#ifndef GPU_HOST_H_
#define GPU_HOST_H_

#include <pthread.h>

#include <deque>

// If you are using "compute-exclusive-thread mode" as a CUDA driver compute mode,
// you need to turn on this flag. This requires linking to Driver API Library.
//#define CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE 1

// cuda
#include "cuda.h"
#include "cuda_runtime.h"

#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
#include "cuda_runtime_api.h"
#endif

// gethostname, getpid
#include <unistd.h>

#ifndef CUDA_CHECK
static void cuda_error_handler(cudaError_t error, const char* srcfile, int linenumber) {
	char hostname[256]; gethostname(hostname, sizeof(hostname));
	int pid = getpid();
	// gethostname
#ifdef CUDA_CHECK_PRINT_RANK
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	printf("cudaError:%s\n%s(PID:%d,Rank:%d) %s(%d)\n", cudaGetErrorString(error), hostname, pid, rank, srcfile, linenumber);
#else
	printf("cudaError:%s\n%s(PID:%d) %s(%d)\n", cudaGetErrorString(error), hostname, pid, srcfile, linenumber);
#endif
	throw "cudaError";
}
#define CUDA_CHECK(method) if( cudaError_t error = (cudaError_t)method  ){ \
		cuda_error_handler(error, __FILE__, __LINE__); \
	 }
#endif

//-------------------------------------------------------------//
// Cuda Stream Manager API
//-------------------------------------------------------------//

class CudaCommand
{
public:
	virtual ~CudaCommand() { }
	virtual bool init() { return true; } // To invalidate this command, return false.
	virtual void send(cudaStream_t stream) = 0;
	virtual void launch_kernel(cudaStream_t stream) = 0;
	virtual void receive(cudaStream_t stream) = 0;
	virtual void complete() = 0;
};

class CudaManageCommand
{
public:
	virtual ~CudaManageCommand() { }
	virtual void process() = 0;
};

class CudaStreamManager
{
	static CudaStreamManager* instance;

	CudaStreamManager(int device_index)
		: device_index_(device_index)
		, cleanup_(false)
		, command_active_(false)
		, suspended_(false)
		, terminated_(false)
	{
		pthread_mutex_init(&thread_sync_, NULL);
		pthread_cond_init(&thread_state_,  NULL);

#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
		CUDA_CHECK(cuInit(0));
		pthread_mutex_init(&context_mutex_, NULL);
#endif
	//	printf("Initializing CUDA ..."); fflush(stdout);
		if(device_index_ != -1) {
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
			CUDA_CHECK(cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, device_index_));
#else
			CUDA_CHECK(cudaSetDevice(device_index_));
#endif
		}
		else {
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
			int dev_count;
			CUDA_CHECK(cudaGetDeviceCount(&dev_count));
			for(int i = 0 ; i < dev_count; ++i) {
				if(cuCtxCreate(&ctx_, CU_CTX_SCHED_AUTO, i))
					continue;
				device_index_ = i;
				break;
			}
			if(device_index_ == -1) {
				printf("cudaError: There is no valid devices.\n");
				throw "cudaError";
			}
#else
			CUDA_CHECK(cudaGetDevice(&device_index_));
#endif
		}
		CUDA_CHECK(cudaGetDeviceProperties(&device_property_, device_index_));
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
		CUDA_CHECK(cuCtxSetCurrent(NULL));
#endif
	//	printf(" complete.\n");

#if 0
		InputCommand empty_cmd; empty_cmd.kind = COMMAND_CUDA; empty_cmd.cuda_cmd = NULL;
		for(int i = 0; i < 128; ++i) for(int p = 0; p < MAX_PRIORITY; ++p) commandQue_[p].push_back(empty_cmd);
		for(int i = 0; i < 128; ++i) for(int p = 0; p < MAX_PRIORITY; ++p) commandQue_[p].pop_front();
#endif

		pthread_create(&thread_, NULL, manager_thread_routine_, this);
	}
	virtual ~CudaStreamManager()
	{
		if(!cleanup_) {
			cleanup_ = true;
			terminated_ = true;
			suspended_ = true;
			command_active_ = true;
			pthread_cond_broadcast(&thread_state_);
			pthread_join(thread_, NULL);
			pthread_mutex_destroy(&thread_sync_);
			pthread_cond_destroy(&thread_state_);
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
			pthread_mutex_destroy(&context_mutex_);
#endif
		}
	}

public:
	static const int MAX_PRIORITY = 4;

	static void initialize_cuda(int device_index) {
		instance = new CudaStreamManager(device_index);
	}
	static void finalize_cuda() {
		delete instance; instance = NULL;
	}
	static CudaStreamManager* get_instance() {
		return instance;
	}
	static void begin_cuda() {
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
		get_instance()->set_context();
#endif
	}
	static void end_cuda() {
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
		get_instance()->release_context();
#endif
	}
	void submit(CudaCommand* command, int priority)
	{
		// create a command
		InputCommand cmd;
		cmd.kind = COMMAND_CUDA;
		cmd.cuda_cmd = command;
		// submit
		bool suspended;
		pthread_mutex_lock(&thread_sync_);
		commandQue_[priority].push_back(cmd);
		command_active_ = true;
		suspended = suspended_;
		pthread_mutex_unlock(&thread_sync_);
		// wake up the managing thread if it is suspended
		if( suspended ) pthread_cond_broadcast(&thread_state_);
	}
	void submit(CudaManageCommand* command, int priority)
	{
		// create a command
		InputCommand cmd;
		cmd.kind = COMMAND_MANAGE;
		cmd.manage_cmd = command;
		// submit
		bool suspended;
		pthread_mutex_lock(&thread_sync_);
		commandQue_[priority].push_back(cmd);
		command_active_ = true;
		suspended = suspended_;
		pthread_mutex_unlock(&thread_sync_);
		// wake up the managing thread if it is suspended
		if( suspended ) pthread_cond_broadcast(&thread_state_);
	}

	int getDeviceIndex() { return device_index_; }
	cudaDeviceProp& getDeviceProp() { return device_property_; }

protected:
	int device_index_;
	cudaDeviceProp device_property_;

	enum COMMAND {
		COMMAND_CUDA = 1,
		COMMAND_MANAGE = 2,
	};

	struct InputCommand {
		COMMAND kind;
		union {
			CudaCommand* cuda_cmd;
			CudaManageCommand* manage_cmd;
		};
	};

	// accessed by user threads as well as gpu managing thread //
	pthread_t thread_;
	pthread_mutex_t thread_sync_;
	pthread_cond_t thread_state_;
	std::deque<InputCommand> commandQue_[MAX_PRIORITY];
	bool cleanup_;
	volatile bool command_active_;
	volatile bool suspended_;
	volatile bool terminated_;

#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
	pthread_mutex_t context_mutex_;
	CUcontext ctx_;
#endif

	enum {
		PHASE_SEND = 0,
		PHASE_KERNEL = 1,
		PHASE_RECEIVE = 2,
		MAX_PHASE = 4,
	};
	// accessed by gpu managing thread only

	static void* manager_thread_routine_(void* pThis) {
		static_cast<CudaStreamManager*>(pThis)->manager_thread_routine();
		return NULL;
	}
	void manager_thread_routine()
	{
		struct StreamSlot {
			CudaCommand* command;
			cudaStream_t stream;
		};
		StreamSlot slot[MAX_PHASE];
		int base_slot = MAX_PHASE;
		int number_active_slot = 0;

#define SEND_SLOT ((base_slot - PHASE_SEND) & (MAX_PHASE - 1))
#define KERNEL_SLOT ((base_slot - PHASE_KERNEL) & (MAX_PHASE - 1))
#define RECEIVE_SLOT ((base_slot - PHASE_RECEIVE) & (MAX_PHASE - 1))

		set_context();
		for(int i = 0; i < MAX_PHASE; ++i) {
			slot[i].command = NULL;
			CUDA_CHECK(cudaStreamCreate(&slot[i].stream));
		}
		release_context();

		// command loop
		while(true) {
			if(slot[SEND_SLOT].command == NULL) {
				if(command_active_) {
					pthread_mutex_lock(&thread_sync_);
					InputCommand cmd;
					while(pop_command(&cmd)) {
						pthread_mutex_unlock(&thread_sync_);
	//					try {
							if(cmd.kind == COMMAND_CUDA) {
								set_context();
								if(cmd.cuda_cmd->init()) {
#if DISABLE_CUDA_CONCCURENT
									cmd.cuda_cmd->send(NULL);
									cmd.cuda_cmd->launch_kernel(NULL);
									cmd.cuda_cmd->receive(NULL);
									cmd.cuda_cmd->complete();
#else
									// start sending
									slot[SEND_SLOT].command = cmd.cuda_cmd;
									slot[SEND_SLOT].command->send(slot[SEND_SLOT].stream);
									++number_active_slot;
									release_context();
									break;
#endif
								}
								release_context();
							}
							else if(cmd.kind == COMMAND_MANAGE) {
								set_context();
								cmd.manage_cmd->process();
								release_context();
							}
							else {
								printf("Unknown Command (%d). Invalid Program!!!\n", (int)cmd.kind);
								throw "Unknown Command. Invalid Program!!!";
							}
	//					}
	//					catch(...) {
	//						printf("cudaManagingProc:error!\n");
	//						throw;
	//					}
						pthread_mutex_lock(&thread_sync_);
					}
					pthread_mutex_unlock(&thread_sync_);
				}
				if(number_active_slot == 0) {
					pthread_mutex_lock(&thread_sync_);
					if(command_active_ == false) {
						suspended_ = true;
						if( terminated_ ) { pthread_mutex_unlock(&thread_sync_); break; }
						pthread_cond_wait(&thread_state_, &thread_sync_);
						suspended_ = false;
					}
					pthread_mutex_unlock(&thread_sync_);
				}
			}

			set_context();
			if(slot[RECEIVE_SLOT].command) {
				// start receiving
				slot[RECEIVE_SLOT].command->receive(slot[RECEIVE_SLOT].stream);
			}
			if(slot[KERNEL_SLOT].command) {
				// start kernel
				slot[KERNEL_SLOT].command->launch_kernel(slot[KERNEL_SLOT].stream);
			}
			// wait for completion of all command
			CUDA_CHECK(cudaThreadSynchronize());
			if(slot[RECEIVE_SLOT].command) {
				slot[RECEIVE_SLOT].command->complete();
				slot[RECEIVE_SLOT].command = NULL;
				--number_active_slot;
			}
			release_context();
			base_slot = ((base_slot + 1) & (MAX_PHASE - 1)) + 4;
		}

#undef SEND_SLOT
#undef KERNEL_SLOT
#undef RECEIVE_SLOT

		set_context();
		for(int i = 0; i < MAX_PHASE; ++i) {
			CUDA_CHECK(cudaStreamDestroy(slot[i].stream));
		}
		release_context();

	}
	bool pop_command(InputCommand* cmd) {
		int i = MAX_PRIORITY;
		while(i-- > 0) {
			if(commandQue_[i].size()) {
				*cmd = commandQue_[i][0];
				commandQue_[i].pop_front();
				return true;
			}
		}
		command_active_ = false;
		return false;
	}
	void set_context() {
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
		pthread_mutex_lock(&context_mutex_);
		CUDA_CHECK(cuCtxSetCurrent(ctx_));
#endif
	}
	void release_context() {
#if CUDA_COMPUTE_EXCLUSIVE_THREAD_MODE
		CUDA_CHECK(cuCtxSetCurrent(NULL));
		pthread_mutex_unlock(&context_mutex_);
#endif
	}
};


#endif /* GPU_HOST_H_ */
