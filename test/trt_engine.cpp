#include "trt_engine.h"

#include <fstream>
#include <sstream>

#include "yolo_utils.h"

static trtLogger gLogger;

using namespace YOLO;

TrtEngine::TrtEngine(){
    _engine = nullptr;
    _runtime = nullptr;
}

TrtEngine::~TrtEngine(){
    destroy();    
}

bool TrtEngine::load(const std::string& engine_file_name){
    if(_runtime || _engine) return true;
    std::string engine_buff;
    std::ifstream engine_file(engine_file_name, std::ios::binary);
    if(!engine_file)
        return false;
    std::ostringstream oss;
    oss << engine_file.rdbuf();
    engine_buff = oss.str();
    engine_file.close();

    _runtime = nvinfer1::createInferRuntime(gLogger);
    if(_runtime == nullptr)
        return false;
    
    _engine = _runtime->deserializeCudaEngine(engine_buff.data(), engine_buff.size());
    if(_engine == nullptr){
        delete _runtime;
        _runtime = nullptr;
        return false;
    }

    return true;
}

void TrtEngine::destroy(){
        for(auto& ptr:_contexts){
        delete ptr;
    }
	if (_engine) {
		delete _engine;
	}
	if(_runtime){
        delete _runtime;
    }
}

nvinfer1::IExecutionContext* TrtEngine::get_context(){
    // TODO: 根据需要决定是否需要回收和再利用context
    if(!_engine){
        return nullptr;
    }
    auto context = _engine->createExecutionContext();
    if(context == nullptr)
        return nullptr;
    _contexts.emplace_back(context);
    return context;
}
