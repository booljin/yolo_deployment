#ifndef __YOLO_TRT_H__
#define __YOLO_TRT_H__
#include "yolo_defines.h"
#include "../src/trt_impl/model_trt.h"
#include "../src/trt_impl/workspace_trt.h"
#include "../src/trt_impl/taskflow_trt.h"
#include "yolo.h"

namespace YOLO{
	struct TRT {};

    template<>
    struct BackendTraits<YOLO::TRT> {
        using Model = YOLO::MODEL::TRT::Model;
        using WorkSpace = YOLO::WORKSPACE::TRT::WorkSpace;
        using TaskFlow = YOLO::TASK::TRT::TaskFlow;
    };

}



#endif // __YOLO_TRT_H__