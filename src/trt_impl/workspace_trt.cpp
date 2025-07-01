#include "workspace_trt.h"

#include <tuple>

#include "model_trt.h"
#include "yolo_defines.h"

using namespace YOLO::WORKSPACE::TRT;

WorkSpace::WorkSpace(){
}

WorkSpace::~WorkSpace(){
    release();
}

bool WorkSpace::create_by(const std::vector<YOLO::MODEL::TRT::Model*>& models){
    if(_type != YOLO::ModelType::MODEL_TYPE_UNKNOWN){
        // 创建过
        return false;
    }
    if(models.size() == 0){
        // 没有模型，操作没有意义
        return false;
    }
    for(auto& model: models){
		if(model->type() != YOLO::ModelType::MODEL_TYPE_TRT){
            // 模型类型不匹配，我只能处理TensorRT模型
            release();
            return false;
        }
        _contexts[model->alias()] = model->create_context_trt();
        //_contexts.emplace_back(static_cast<Model*>(model)->create_context_trt());
    }
    _type = YOLO::ModelType::MODEL_TYPE_TRT;
    return true;
}

nvinfer1::IExecutionContext* WorkSpace::at(const std::string& alias){
    if(_contexts.find(alias) != _contexts.end())
        return _contexts[alias];
    else
        return nullptr;
}

void WorkSpace::release(){
    for(auto& context: _contexts){
        if(context.second != nullptr){
            delete context.second;
        }
    }
    _contexts.clear();
    _type = YOLO::ModelType::MODEL_TYPE_UNKNOWN;
}