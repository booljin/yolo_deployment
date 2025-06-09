#ifndef __YOLO_WORKSPACE_BASE_H__
#define __YOLO_WORKSPACE_BASE_H__

#include <vector>
#include <memory>
#include "yolo_defines.h"

namespace YOLO{

class ModelAny;

class WorkSpaceAny{
public:
    virtual ~WorkSpaceAny() = default;
public:
    virtual bool create_by(const std::vector<std::shared_ptr<ModelAny>>& models) = 0;
    virtual void* context_at(int index) = 0;
protected:
    ModelType _type = ModelType::MODEL_TYPE_UNKNOWN;
};

}
#endif // __YOLO_WORKSPACE_BASE_H__