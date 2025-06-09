#ifndef __YOLO_UTILS_H__
#define __YOLO_UTILS_H__

#include <spdlog/spdlog.h>
#include <NvInferRuntime.h>
#include <stdexcept>
#include <iostream>
#define CHECK(status) \
do\
{\
    auto ret = (status);\
    if (ret != 0)\
    {\
        std::cerr << "Cuda failure: " << ret << std::endl;\
        abort();\
    }\
} while (0)

class trtLogger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept override
	{
		if (severity == Severity::kINTERNAL_ERROR)
			SPDLOG_CRITICAL(msg);
		else if (severity == Severity::kERROR)
			SPDLOG_ERROR(msg);
		else if (severity == Severity::kWARNING)
			SPDLOG_WARN(msg);
		else if (severity == Severity::kINFO)
			SPDLOG_INFO(msg);
		else
			SPDLOG_DEBUG(msg);
	}
};

namespace YOLO {
	class WorkSpace {
	public:
		WorkSpace(nvinfer1::IExecutionContext* ctx) :_context(ctx) {}
		WorkSpace() :_context(nullptr) {}
	public:
		inline nvinfer1::IExecutionContext* context() { return _context; }
		inline void setContext(nvinfer1::IExecutionContext* ctx) { _context = ctx; }
	private:
		nvinfer1::IExecutionContext* _context = nullptr;
	};

	enum MemcpyKind {
		MemcpyDeviceToHost = 1,
		MemcpyHostToDevice = 2,
	};
	template<class T>
	class CudaMemory {
	public:
		CudaMemory& malloc(size_t len) {
			release();
			_len = len;
			return *this;
		}
		/*T* malloc_cpu(size_t len) {
			if (_len != 0) {
				if (_len == len){
					if (_host == nullptr) {
						CHECK(cudaMallocHost((void**)&_host, _len * sizeof(T)));
					}
					return _host;
				}
				else
					return nullptr;
			}
			_len = len;
			CHECK(cudaMallocHost((void**)&_host, _len * sizeof(T)));
			return _host;
		}
		T* malloc_gpu(size_t len) {
			if (_len != 0) {
				if (_len == len) {
					if (_device == nullptr) {
						CHECK(cudaMalloc((void**)&_device, _len * sizeof(T)));
					}
					return _device;
				}
				else
					return nullptr;
			}
			_len = len;
			CHECK(cudaMalloc((void**)&_host, _len * sizeof(T)));
			return _device;
		}*/
		T* cpu() {
			if (_len > 0 && _host == nullptr) {
				CHECK(cudaMallocHost((void**)&_host, _len * sizeof(T)));
			}
			return _host;
		}
		T* gpu() {
			if (_len > 0 && _device == nullptr) {
				CHECK(cudaMalloc((void**)&_device, _len * sizeof(T)));
			}
			return _device;
		}

		void memcpy_async(int kind, cudaStream_t& stream) {
			if (kind == MemcpyDeviceToHost) {
				if (_device == nullptr)
					return;
				CHECK(cudaMemcpyAsync(cpu(), _device, _len * sizeof(T), cudaMemcpyDeviceToHost, stream));
			} else {
				if (_host == nullptr)
					return;
				CHECK(cudaMemcpyAsync(gpu(), _host, _len * sizeof(T), cudaMemcpyHostToDevice, stream));
			}
		}

		inline void memcpy_to_cpu(cudaStream_t& stream) {
			if (_len == 0)
				return;
			memcpy_async(MemcpyDeviceToHost, stream);
		}
		inline void memcpy_to_gpu(cudaStream_t& stream) {
			if (_len == 0)
				return;
			memcpy_async(MemcpyHostToDevice, stream);
		}

		inline void memcpy_to_cpu_sync(cudaStream_t& stream) {
			if (_len == 0)
				return;
			memcpy_async(MemcpyDeviceToHost, stream);
			CHECK(cudaStreamSynchronize(stream));

		}
		inline void memcpy_to_gpu_sync(cudaStream_t& stream) {
			if (_len == 0)
				return;
			memcpy_async(MemcpyHostToDevice, stream);
			CHECK(cudaStreamSynchronize(stream));
		}


		size_t len() { return _len; }

		void release() {
			if (_len > 0) {
				if (_device != nullptr) {
					CHECK(cudaFree(_device));
					_device = nullptr;
				}
				if (_host != nullptr) {
					CHECK(cudaFreeHost(_host));
					_host = nullptr;
				}
				_len = 0;
			}
		}

		~CudaMemory() {
			release();
		}
	private:
		size_t _len = 0;
		T* _device = nullptr;
		T* _host = nullptr;
	};
}

#endif //__YOLO_UTILS_H__