#ifndef __YOLO_WORKSPACE_TRT_H__
#define __YOLO_WORKSPACE_TRT_H__

#include "workspace_base.h"
#include <NvInfer.h>

namespace YOLO{
namespace TRT{

class WorkSpace : public WorkSpaceAny {
public:
    WorkSpace();
    virtual ~WorkSpace() override;
public:
    bool create_by(const std::vector<std::shared_ptr<ModelAny>>& models) override;
    void* context_at(int index) override;
    nvinfer1::IExecutionContext* at(int index);
private:
    void release();

private:
    std::vector<nvinfer1::IExecutionContext*> _contexts;
};

}
}

#endif // __YOLO_WORKSPACE_TRT_H__