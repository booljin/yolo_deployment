#ifndef __YOLO_TRT_UTILS_H__
#define __YOLO_TRT_UTILS_H__ 

#include <spdlog/spdlog.h>
#include <NvInferRuntime.h>
#include "yolo_defines.h"
#include <fmt/format.h>

#define CUDA_CHECK(status) \
do\
{\
    auto ret = (status);\
    if (ret != 0)\
    {\
		throw YOLO::YoloException(fmt::format("Cuda failure: {}", (int)ret));\
    }\
} while (0)



namespace YOLO {
namespace UTILS{
namespace TRT{

class Logger : public nvinfer1::ILogger
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


enum MemcpyKind {
    MemcpyDeviceToHost = 1,
    MemcpyHostToDevice = 2,
};
template<class T>
class CudaMemory {
public:
	CudaMemory() = default;
	CudaMemory(CudaMemory&& s) {
		_host = s._host;
		s._host = nullptr;
		_device = s._device;
		s._device = nullptr;
	}

    CudaMemory& malloc(size_t len) {
        release();
        _len = len;
        return *this;
    }

    T* cpu() {
        if (_len > 0 && _host == nullptr) {
			CUDA_CHECK(cudaMallocHost((void**)&_host, _len * sizeof(T)));
			CUDA_CHECK(cudaMemset((void*)_host, 0, _len * sizeof(T)));
        }
        return _host;
    }
    T* gpu() {
        if (_len > 0 && _device == nullptr) {
			CUDA_CHECK(cudaMalloc((void**)&_device, _len * sizeof(T)));
			CUDA_CHECK(cudaMemset((void*)_device, 0, _len * sizeof(T)));
        }
        return _device;
    }

    void memcpy_async(int kind, cudaStream_t& stream) {
        if (kind == MemcpyDeviceToHost) {
            if (_device == nullptr)
                return;
			CUDA_CHECK(cudaMemcpyAsync(cpu(), _device, _len * sizeof(T), cudaMemcpyDeviceToHost, stream));
        } else {
            if (_host == nullptr)
                return;
			CUDA_CHECK(cudaMemcpyAsync(gpu(), _host, _len * sizeof(T), cudaMemcpyHostToDevice, stream));
        }
    }
    void memcpy_sync(int kind) {
        if (kind == MemcpyDeviceToHost) {
            if (_device == nullptr)
                return;
			CUDA_CHECK(cudaMemcpy(cpu(), _device, _len * sizeof(T), cudaMemcpyDeviceToHost));
        } else {
            if (_host == nullptr)
                return;
			CUDA_CHECK(cudaMemcpy(gpu(), _host, _len * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    inline void memcpy_to_cpu_sync() {
        if (_len == 0)
            return;
        memcpy_sync(MemcpyDeviceToHost);
    }
    inline void memcpy_to_gpu_sync() {
        if (_len == 0)
            return;
        memcpy_sync(MemcpyHostToDevice);
    }

    inline void memcpy_to_cpu_async(cudaStream_t& stream) {
        if (_len == 0)
            return;
        memcpy_async(MemcpyDeviceToHost, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));

    }
    inline void memcpy_to_gpu_async(cudaStream_t& stream) {
        if (_len == 0)
            return;
        memcpy_async(MemcpyHostToDevice, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    size_t len() { return _len; }

    void release() {
        if (_len > 0) {
            if (_device != nullptr) {
				CUDA_CHECK(cudaFree(_device));
                _device = nullptr;
            }
            if (_host != nullptr) {
				CUDA_CHECK(cudaFreeHost(_host));
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

void dump_preprocess_img(CudaMemory<float>& img, const std::string& path);

}}}
#endif // __YOLO_TRT_UTILS_H__