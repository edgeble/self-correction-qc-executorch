# From the trained .pt to the sealed .pte (and the A/B pair)

Everything here runs on a laptop in docker; the board never quantizes.

1. **Image** `tools/Dockerfile.executorch-x86`: Ubuntu 22.04, Python 3.10, torch 2.14 (CPU),
   torchao 0.18, executorch 1.5.1, ultralytics 8.3.156, plus the patch that adds the Dragonwing
   IQ8 to ExecuTorch's chipset tables (`QCS8275 = 82`, Hexagon V75: `qc_schema.py`, `utils.py`
   name maps, `qc_compiler_spec.fbs`). Upstream ExecuTorch does not list this SoC yet.
2. **QAIRT SDK** extracted on the laptop, mounted at `/qairt`; its x86 HTP libraries build the
   delegate at export.
3. **Calibration** `tools/make_calib.py <dir> <video>...`: 70 frames of the rig, letterboxed to
   640x640, RGB 0..1, float32 NCHW.
4. **Sealed export** `tools/export_pte.py --pt 9v-bat-2.pt --soc QCS8275 --calib <dir>
   --act-observer Histogram --out <name>`: Ultralytics load, the Detect head replaced by the
   nine-output form (per scale: box DFL logits [64], class sigmoid [nc], score sum [1]),
   `torch.export`, PT2E quantization (8-bit activations, Histogram observer; per-channel 8-bit
   weights), `to_edge_transform_and_lower_to_qnn`, `to_executorch`. Output `.pte` (3.1 MB): one
   QNN delegate, float32 input [1,3,640,640], float32 outputs. The script prints float / int8 /
   ONNX detections side by side for a parity check (`model/export-9v2-sealed-QCS8275.log`).
Why the Histogram observer: the default moving-average min-max observer dropped class scores from
0.87 to 0.62 on the calibration frames; Histogram keeps them at 0.79-0.86 with the same boxes.
