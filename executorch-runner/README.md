# The ExecuTorch runner (C++)

The part of edgeble-infer written for this hackathon, in the open. It sits behind the engine's
runner interface (`yolo_runner.h`: init / infer_slot / release) and is picked when the model file
ends in `.pte`.

| file | role |
|---|---|
| `qcom_runner.h` | the runner seam: `init(model, impl, w, h)`, `run(impl, rgb, outs, n)`, `release`, optional `aux` (the float tail's inputs) |
| `yolo_runner_qcom.cpp` | the dispatcher: letterbox on the CPU, picks the runner by file name, hands the outputs to the post-processor, can dump frames with the head's inputs and detections (`EDGEBLE_FEAT_DUMP`) |
| `yolo_runner_executorch.cpp` | ExecuTorch `Module` API: loads `<name>.pte`, or `<name>-A.pte` with `<name>-B.pte` and `<name>-B.ptd` beside it, chains A (QNN delegate on the Hexagon) into B (float, CPU), returns `[box, cls, sum] x 3` as `ed_tensor_t` |
| `yolo11_postprocess.h` | the backend-neutral tensor description the post-processor consumes |

Links against `executorch_core`, `extension_module_static`, `extension_tensor`,
`extension_data_loader`, `extension_flat_tensor`, `portable_ops_lib`, `quantized_ops_lib` and
`qnn_executorch_backend` from an ExecuTorch install built with `-DEXECUTORCH_BUILD_QNN=ON`
(`tools/build-aarch64.sh`). The QNN libraries are dlopened by name from `EDGEBLE_QNN_LIBS` first,
so the backend finds the same copies; `ADSP_LIBRARY_PATH` points at the Hexagon skel.
