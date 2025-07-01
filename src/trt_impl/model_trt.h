#ifndef __YOLO_MODEL_TRT_H__
#define __YOLO_MODEL_TRT_H__
#include "model_base.h"
#include <NvInfer.h>
#include <list>
#include <string>
#include <filesystem>
#include <thread>

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
};


}}}


#endif // __YOLO_MODEL_TRT_H__