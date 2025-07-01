#ifndef __YOLO_WORKSPACE_TRT_H__
#define __YOLO_WORKSPACE_TRT_H__

#include "workspace_base.h"
#include <NvInfer.h>
#include <map>
#include "model_trt.h"

namespace YOLO{
namespace WORKSPACE{
namespace TRT{

class WorkSpace : public YOLO::WORKSPACE::Any {
public:
    WorkSpace();
    virtual ~WorkSpace() override;
public:
    bool create_by(const std::vector<YOLO::MODEL::TRT::Model*>& models);
    nvinfer1::IExecutionContext* at(const std::string& alias);
private:
    void release();

private:
    std::map<std::string, nvinfer1::IExecutionContext*> _contexts;
};

}}}

#endif // __YOLO_WORKSPACE_TRT_H__