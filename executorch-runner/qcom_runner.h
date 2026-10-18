/*
 * The runners behind lib/model/yolo_runner.h on Qualcomm boards. The
 * dispatcher (yolo_runner_qcom.cpp) owns the CPU side, the BGR swap, the
 * letterbox, the post-processor and the result mapping, and picks the
 * runner by the model file:
 *
 *   .pte            ExecuTorch program lowered to the QNN backend
 *                   (yolo_runner_executorch.cpp, when built in); a
 *                   `<name>-A.pte` with a `<name>-B.pte` beside it is the
 *                   pair: A on the NPU, B on the CPU, weights in `<name>-B.ptd`
 *   anything else   QNN HTP context binary (yolo_runner_qnn.cpp)
 *
 * A runner takes the letterboxed RGB888 frame and returns the graph's
 * outputs described as ed_tensor_t, valid until its next run().
 */
#ifndef EDGEBLE_QCOM_RUNNER_H
#define EDGEBLE_QCOM_RUNNER_H

#include "model/yolo11/yolo11_postprocess.h"

#include <stdint.h>

#include <string>

struct qcom_runner {
	const char *name;
	/* loads the model; reports the input size the frame has to be letterboxed to */
	int (*init)(const char *model_path, void **impl, int *model_w, int *model_h);
	/* rgb: RGB888 HWC, model_w x model_h */
	int (*run)(void *impl, const uint8_t *rgb, const ed_tensor_t **outs, int *n_outs);
	void (*release)(void *impl);
	/* optional: the float tail's input tensors of the last run() */
	int (*aux)(void *impl, const ed_tensor_t **feats, int *n_feats);
};

extern const qcom_runner qcom_runner_qnn;
#if EDGEBLE_HAS_EXECUTORCH
extern const qcom_runner qcom_runner_executorch;
#endif

/* where the QNN runtime libraries are: EDGEBLE_QNN_LIBS, else the model's
 * directory, else the system paths; the bare name when none has it */
std::string qcom_find_lib(const char *model_path, const char *name);
std::string qcom_dir_of(const char *path);
/* the Hexagon side loads its skel library through fastrpc, which looks in
 * ADSP_LIBRARY_PATH; point it at the runtime directory unless it is set */
void qcom_set_adsp_path(const std::string &libdir);

#endif /* EDGEBLE_QCOM_RUNNER_H */
