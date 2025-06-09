#include "model_trt.h"

#include <fstream>
#include <sstream>

#include "yolo_defines.h"
#include "yolo_trt_utils.h"
static YOLO::TRT::Logger gLogger;

using namespace YOLO::TRT;

class TRT_Runtime{
public:
    ~TRT_Runtime(){
        if(_runtime){
            delete _runtime;
        }
    }
    static nvinfer1::IRuntime* getRuntime(){
        static TRT_Runtime instance;
        if(!instance._runtime){
            instance._runtime = nvinfer1::createInferRuntime(gLogger);
        }
        return instance._runtime;
    }
private:
    TRT_Runtime() = default;
    nvinfer1::IRuntime* _runtime = nullptr;
};


Model::Model() : ModelAny(){
    _engine = nullptr;
}

Model::~Model(){
    destroy();    
}

Model::Model(Model&& model){
    _engine = model._engine;
    _type = model._type;
    _status = model._status;
    _task_type = model._task_type;
    model._engine = nullptr;
    _status = YOLO::ModelAny::STATUS_Ready;
}

bool Model::load(const std::string& engine_file_name){
    if(_type != YOLO::ModelType::MODEL_TYPE_UNKNOWN) return true;
    std::string engine_buff;
    std::ifstream engine_file(engine_file_name, std::ios::binary);
    if(!engine_file)
        return false;
    std::ostringstream oss;
    oss << engine_file.rdbuf();
    engine_buff = oss.str();
    engine_file.close();

    auto runtime = TRT_Runtime::getRuntime();
    
    _engine = runtime->deserializeCudaEngine(engine_buff.data(), engine_buff.size());
    if(_engine == nullptr){
        return false;
    }
    _type = YOLO::ModelType::MODEL_TYPE_TRT;

    // 读取模型维度，判断任务类型
    auto dims = _engine->getTensorShape("output1");
    if(dims.nbDims == 4){
        // 分割任务有两个输出，一个是检测框，一个是mask。其中mask有4个维度：batch,maskdim,h,w
        _task_type = YOLO::ModelAny::TASK_SEGMENT;
    } else {
        auto dims_0 = _engine->getTensorShape("output0");
        if(dims_0.nbDims == 3){
            // 检测任务有3个维度：batch，bbox， box_count
            _task_type = YOLO::ModelAny::TASK_DETECT;
        } else {
            // 分类任务有2个维度：batch， class_count
            _task_type = YOLO::ModelAny::TASK_CLASSIFY;
        }
    }
    return true;
}

void Model::destroy(){
	if (_engine) {
		delete _engine;
        _engine = nullptr;
	}
}

nvinfer1::IExecutionContext* Model::create_context_trt(){
    if(!_engine){
        return nullptr;
    }
    if(_status != YOLO::ModelAny::STATUS_Ready){
        // 模型未准备好，不能创建上下文
        return nullptr;
    }
    auto context = _engine->createExecutionContext();
    return context;
}

void* Model::create_context(){
    return create_context_trt();
}