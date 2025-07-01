#ifndef __YOLO_WORKSPACE_BASE_H__
#define __YOLO_WORKSPACE_BASE_H__

#include <vector>
#include <memory>
#include "yolo_defines.h"

namespace YOLO{
namespace WORKSPACE{

class Any{
public:
    virtual ~Any() = default;
protected:
    YOLO::ModelType _type = YOLO::ModelType::MODEL_TYPE_UNKNOWN;
};

}}
#endif // __YOLO_WORKSPACE_BASE_H__