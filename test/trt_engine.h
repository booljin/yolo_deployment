#ifndef __TRT_ENGINE_H__
#define __TRT_ENGINE_H__ 

#include <NvInfer.h>
#include <list>

namespace YOLO{

class TrtEngine{
public:
    TrtEngine();
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine operator=(const TrtEngine&) = delete;
    ~TrtEngine();

public:
    bool load(const std::string& engine_file);
    void destroy();
    nvinfer1::IExecutionContext* get_context();
	inline nvinfer1::ICudaEngine* engine() { return _engine; }
private:
    nvinfer1::IRuntime* _runtime;
    nvinfer1::ICudaEngine* _engine;
    std::list<nvinfer1::IExecutionContext*> _contexts;
};


}

#endif//__TRT_ENGINE_H__