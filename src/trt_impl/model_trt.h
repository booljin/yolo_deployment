#ifndef __YOLO_MODEL_TRT_H__
#define __YOLO_MODEL_TRT_H__
#include "model_base.h"
#include <NvInfer.h>
#include <list>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>

namespace YOLO{
namespace MODEL{
namespace TRT{

class Model : public YOLO::MODEL::Any {
public:
	Model(const std::string& alias);
    ~Model() override;

public:
    bool load(const std::string& file_name);
    void destroy();

    void* create_context() override;
    nvinfer1::IExecutionContext* create_context_trt();
    inline nvinfer1::ICudaEngine* engine() { return _engine; }

public:
	YOLO::TaskType task_type() override { std::lock_guard<std::mutex>lock(_task_type_mtx);  return _task_type; }

private:
    bool is_engine_file_valid(const std::filesystem::path& onnx, const std::filesystem::path& engine);
	void convert_from_onnx(const std::filesystem::path& onnx, const std::filesystem::path& engine);
    bool build_engine(const char* buffer, size_t size);
private:
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model& operator=(Model&& engine) = delete;
private:
    nvinfer1::ICudaEngine* _engine;
private:
    std::thread _convert_thread;
	std::mutex _task_type_mtx;	// 如果模型正在异步转换中，task_type会在转换中途识别。这里需要加锁，以保证后续模型能排队等到前置模型正确识别出类型
};


}}}


#endif // __YOLO_MODEL_TRT_H__