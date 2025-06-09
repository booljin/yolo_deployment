#include "workspace_trt.h"

#include "model_trt.h"
#include "yolo_defines.h"

using namespace YOLO::TRT;

WorkSpace::WorkSpace(){
}

WorkSpace::~WorkSpace(){
    release();
}

bool WorkSpace::create_by(const std::vector<std::shared_ptr<YOLO::ModelAny>>& models){
    if(_type != ModelType::MODEL_TYPE_UNKNOWN){
        // 创建过
        return false;
    }
    if(models.size() == 0){
        // 没有模型，操作没有意义
        return false;
    }
    for(auto& model: models){
        if(model->type() != ModelType::MODEL_TYPE_TRT){
            // 模型类型不匹配，我只能处理TensorRT模型
            release();
            return false;
        }
        _contexts.emplace_back(static_cast<Model*>(model.get())->create_context_trt());
    }
    _type = ModelType::MODEL_TYPE_TRT;
    return true;
}

void* WorkSpace::context_at(int index){
    return at(index);
}

nvinfer1::IExecutionContext* WorkSpace::at(int index){
    return _contexts[index];
}

void WorkSpace::release(){
    for(auto& context: _contexts){
        if(context != nullptr){
            delete context;
        }
    }
    _contexts.clear();
    _type = ModelType::MODEL_TYPE_UNKNOWN;
}