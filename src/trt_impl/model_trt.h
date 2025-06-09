#ifndef __YOLO_MODEL_TRT_H__
#define __YOLO_MODEL_TRT_H__
#include "model_base.h"
#include <NvInfer.h>
#include <list>
#include <string>

namespace YOLO{
namespace TRT{

class Model : public ModelAny {
public:
    Model();
    ~Model() override;

public:
    Model(Model&& model);

public:
    bool load(const std::string& engine_file_name);
    void destroy();

    void* create_context() override;
    nvinfer1::IExecutionContext* create_context_trt();
    inline nvinfer1::ICudaEngine* engine() { return _engine; }


private:
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model& operator=(Model&& engine) = delete;
private:
    nvinfer1::ICudaEngine* _engine;
};


}
}


#endif // __YOLO_MODEL_TRT_H__