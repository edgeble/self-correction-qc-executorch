#!/usr/bin/env python3
# The detector's own backbone as the judge's embedder: 9v-bat-2.pt layers 0..SPPF, global average
# pool -> one vector per crop. Input float32 NCHW 1x3xSxS RGB 0..1 (the crop resized to SxS), no
# mean/std (YOLO convention). Compiled for the Hexagon: int8 (PT2E, Histogram observer) by default.
import argparse, glob, os, time, numpy as np, torch, torch.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument('--pt', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--soc', default='QCS8275'); ap.add_argument('--size', type=int, default=224); ap.add_argument('--fp16', action='store_true')
ap.add_argument('--calib', help='dir of letterboxed float32 NCHW 640 .raw frames: random crops are taken for calibration')
a = ap.parse_args(); os.environ.setdefault('QNN_SDK_ROOT', '/qairt'); torch.manual_seed(0)
from ultralytics import YOLO
net = YOLO(a.pt).model.float().eval(); net.fuse()
layers = list(net.model)
sppf = max(i for i, m in enumerate(layers) if type(m).__name__ == 'SPPF')
class Embed(nn.Module):
    def __init__(self, ls): super().__init__(); self.ls = nn.ModuleList(ls)
    def forward(self, x):
        for m in self.ls: x = m(x)
        return x.mean(dim=(2, 3))          # [1, C]
model = Embed(layers[:sppf + 1]).eval()
for p in model.parameters(): p.requires_grad_(False)
S = a.size; x = torch.rand(1, 3, S, S)
with torch.no_grad(): y = model(x)
print('backbone layers 0..%d (SPPF), embedding dim %d' % (sppf, y.shape[1]))
from executorch.backends.qualcomm.utils.qnn_sdk_setup import setup_qnn_sdk, disable_mkldnn_on_amd
setup_qnn_sdk(); disable_mkldnn_on_amd()
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset, QnnExecuTorchBackendType
from executorch.backends.qualcomm.utils.utils import generate_htp_compiler_spec, generate_qnn_executorch_compiler_spec, to_edge_transform_and_lower_to_qnn, get_qnn_context_binary_alignment
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
soc = getattr(QcomChipset, a.soc); t0 = time.time(); src = model
if not a.fp16:
    from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
    import torchao.quantization.pt2e as pt2e
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    cap = torch.export.export(model, (x,), strict=False).module()
    q = QnnQuantizer(backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=soc)
    q.set_default_quant_config(QuantDtype.use_8a8w, is_qat=False, is_conv_per_channel=True, is_linear_per_channel=False, is_embedding_per_channel=False, act_observer=pt2e.HistogramObserver, act_symmetric=False, eps=None); q.set_submodule_qconfig_list([])
    ann = prepare_pt2e(cap, q); n = 0
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(a.calib, '*.raw')))[:60]:
            fr = torch.from_numpy(np.fromfile(f, dtype=np.float32).reshape(1, 3, 640, 640))
            for _ in range(2):   # two random crops of plausible battery size per frame
                w = int(torch.randint(120, 320, (1,))); h = int(torch.randint(120, 400, (1,))); x0 = int(torch.randint(0, 640 - w, (1,))); y0 = int(torch.randint(0, 640 - h, (1,)))
                crop = torch.nn.functional.interpolate(fr[:, :, y0:y0 + h, x0:x0 + w], size=(S, S), mode='bilinear', align_corners=False)
                ann(crop); n += 1
    src = convert_pt2e(ann); print('int8 calibrated on', n, 'crops')
spec = generate_qnn_executorch_compiler_spec(soc_model=soc, backend_options=generate_htp_compiler_spec(use_fp16=a.fp16))
edge = to_edge_transform_and_lower_to_qnn(src, (x,), spec)
ops = {}
for nd in edge.exported_program().graph.nodes:
    if nd.op == 'call_function': k = str(nd.target); ops[k] = ops.get(k, 0) + 1
print('edge graph', ops)
prog = edge.to_executorch(config=ExecutorchBackendConfig(memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True), segment_alignment=get_qnn_context_binary_alignment()))
with open(a.out, 'wb') as f: prog.write_to_file(f)
print('pte', a.out, os.path.getsize(a.out), 'bytes', f'{time.time()-t0:.0f}s')
