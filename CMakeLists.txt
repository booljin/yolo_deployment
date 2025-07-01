cmake_minimum_required(VERSION 3.24)
project(yolo_deployment LANGUAGES CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS On)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")

#设置C++版本
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#设置警告等级
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

Add_Definitions(-DUNICODE -D_UNICODE -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)

set(TensorRT_DIR "E:/work/print/flower/thirdpart/TensorRT-10.10.0.31/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV)
find_package(TensorRT 10 REQUIRED COMPONENTS nvinfer nvonnxparser)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)

set(ONNXRuntime_DIR "E:/work/print/flower/thirdpart/onnxruntime-win-x64-1.22.0/cmake")
find_package(ONNXRuntime REQUIRED)

set(export
	include/yolo.h
)
source_group("export" FILES ${base})

set(base
	include/model_base.h
	include/task_base.h
	include/workspace_base.h
	include/yolo_defines.h
	include/yolo_scheduler.h
	src/task_utils.cpp
	#src/yolo_scheduler.cpp
)
source_group("base" FILES ${base} ${export})

set(tensorrt
	src/trt_impl/yolo_trt_utils.h
	src/trt_impl/yolo_trt_utils.cpp
	src/trt_impl/model_trt.h
	src/trt_impl/model_trt.cpp
	src/trt_impl/workspace_trt.h
	src/trt_impl/workspace_trt.cpp
	src/trt_impl/task_trt.h
	src/trt_impl/taskflow_trt.h
	src/trt_impl/taskflow_trt.cpp
	src/trt_impl/task_classify_trt.h
	src/trt_impl/task_classify_trt.cpp
	src/trt_impl/task_detect_trt.h
	src/trt_impl/task_detect_trt.cpp
	src/trt_impl/task_segment_trt.h
	src/trt_impl/task_segment_trt.cpp

	include/impl/yolo_trt.h
)
source_group("tensorrt" FILES ${tensorrt})

set(cuda_operator
	src/trt_impl/CUDA_operator/preprocess.cu
	src/trt_impl/CUDA_operator/postprocess_classify.cu
	src/trt_impl/CUDA_operator/postprocess_detect.cu
	src/trt_impl/CUDA_operator/postprocess_segment.cu
)
source_group("tensorrt/cuda_operator" FILES ${cuda_operator})

add_library(trt STATIC
    ${base}
	${export}
	${tensorrt}
	${cuda_operator}
)
add_library(${PROJECT_NAME}::trt ALIAS trt)


target_include_directories(trt
	PUBLIC include
	PUBLIC include/impl
	PRIVATE src
)

target_link_libraries(trt
	TensorRT::nvinfer

	${CUDA_cudart_LIBRARY}
	${CUDA_LIBRARIES}
	${OpenCV_LIBS}

	spdlog::spdlog
	fmt::fmt
)

add_executable(test
	test/main.cpp
	test/logging.h
	test/postprocess_classification.cu
	test/postprocess_detection.cu
	test/postprocess_segmentation.cu
	test/preprocess_t.cu
	test/preprocess_cpu.cpp
	test/trt_engine.cpp
	test/trt_engine.h
	test/yolo_base.cpp
	test/yolo_base.h
	test/yolo_classification_trt.cpp
	test/yolo_classification_trt.h
	test/yolo_detection_trt.cpp
	test/yolo_detection_trt.h
	test/yolo_segmentation_trt.cpp
	test/yolo_segmentation_trt.h
	test/yolo_utils.h
	test/yolo_struct_def.h
)

target_include_directories(test
	PRIVATE test
)

target_link_libraries(test
	${PROJECT_NAME}::trt
	${OpenCV_LIBS}
	${CUDA_cudart_LIBRARY}
	${CUDA_LIBRARIES}
	spdlog::spdlog
	TensorRT::nvinfer
	TensorRT::nvonnxparser
	ONNXRuntime
)

