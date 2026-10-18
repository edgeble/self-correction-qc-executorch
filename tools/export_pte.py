#!/usr/bin/env python3
# fintune.pt -> ExecuTorch .pte for Qualcomm HTP, edgeble-infer 9-output YOLO11 head
# (per scale: box 64 raw DFL logits, class sigmoid, score sum clipped 0..1), int8 PTQ
# via the Qualcomm PT2E quantizer or fp16 (--fp16).  Runs in executorch-x86:1.5.1
# with QAIRT at /qairt.
import argparse, glob, json, os, sys, time, types
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('--pt', required=True)
ap.add_argument('--out', required=True, help='output path without .pte')
ap.add_argument('--soc', default='QCS9100', help='QcomChipset name (QCS9100 = Q900 family)')
ap.add_argument('--imgsz', type=int, default=640)
ap.add_argument('--calib', help='dir of float32 NCHW RGB 0..1 .raw frames (letterboxed)')
ap.add_argument('--n', type=int, default=70)
ap.add_argument('--fp16', action='store_true', help='no quantization, HTP fp16')
ap.add_argument('--ref', help='9-output ONNX of the same model for a parity check')
ap.add_argument('--check', type=int, default=3, help='frames to compare')
ap.add_argument('--online-prepare', action='store_true', help='ship the QNN graph, compile on the device at load (no SoC-specific context)')
ap.add_argument('--dtype', default='8a8w', help='QuantDtype: 8a8w, 16a8w, 16a16w')
ap.add_argument('--act-observer', default='MovingAverageMinMax', help='MovingAverageMinMax, MinMax, Histogram, MovingAveragePerChannelMinMax')
args = ap.parse_args()
os.environ.setdefault('QNN_SDK_ROOT', '/qairt')
torch.manual_seed(0)

from ultralytics import YOLO
net = YOLO(args.pt).model.float().eval()
net.fuse()
names = [net.names[i] for i in range(len(net.names))]
det = net.model[-1]
def nine(self, x):
    outs = []
    for i in range(self.nl):
        box = self.cv2[i](x[i])
        cls = torch.sigmoid(self.cv3[i](x[i]))
        outs += [box, cls, torch.clamp(cls.sum(1, keepdim=True), 0, 1)]
    return outs
det.forward = types.MethodType(nine, det)
class Wrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return tuple(self.m(x))
model = Wrap(net).eval()
for p in model.parameters(): p.requires_grad_(False)

H = W = args.imgsz
frames = []
if args.calib:
    for f in sorted(glob.glob(os.path.join(args.calib, '*.raw')))[:args.n]:
        a = np.fromfile(f, dtype=np.float32)
        assert a.size == 3 * H * W, f'{f}: {a.size} != 3*{H}*{W}'
        frames.append(torch.from_numpy(a.reshape(1, 3, H, W)))
sample = frames[0] if frames else torch.rand(1, 3, H, W)
with torch.no_grad():
    ref_outs = [o.clone() for o in model(sample)]
print('outputs:', [tuple(o.shape) for o in ref_outs])

from executorch.backends.qualcomm.utils.qnn_sdk_setup import setup_qnn_sdk, disable_mkldnn_on_amd
setup_qnn_sdk(); disable_mkldnn_on_amd()
from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import QnnExecuTorchBackendType
import torchao.quantization.pt2e as pt2e
from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
from executorch.backends.qualcomm.utils.utils import (generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec, to_edge_transform_and_lower_to_qnn,
    get_qnn_context_binary_alignment)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

soc = getattr(QcomChipset, args.soc)
t0 = time.time()
if args.fp16:
    src = model
else:
    assert frames, '--calib needed for int8'
    captured = torch.export.export(model, (sample,), strict=False).module()
    # same as export_utils.make_quantizer (not imported: it pulls in pydot via the debugger)
    quantizer = QnnQuantizer(backend=QnnExecuTorchBackendType.kHtpBackend, soc_model=soc)
    act_obs = getattr(pt2e, args.act_observer + 'Observer')
    quantizer.set_default_quant_config(getattr(QuantDtype, 'use_' + args.dtype), is_qat=False, is_conv_per_channel=True,
        is_linear_per_channel=False, is_embedding_per_channel=False,
        act_observer=act_obs, act_symmetric=False, eps=None)
    print('quant config:', args.dtype, 'act observer', args.act_observer)
    quantizer.set_submodule_qconfig_list([])
    annotated = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for i, d in enumerate(frames):
            annotated(d)
    src = convert_pt2e(annotated)
    print(f'PTQ calibrated on {len(frames)} frames in {time.time()-t0:.0f}s')

spec = generate_qnn_executorch_compiler_spec(soc_model=soc, online_prepare=args.online_prepare,
        backend_options=generate_htp_compiler_spec(use_fp16=args.fp16))
edge = to_edge_transform_and_lower_to_qnn(src, (sample,), spec)
g = edge.exported_program().graph
ops = {}
for n in g.nodes:
    if n.op == 'call_function':
        k = str(n.target).split('.')[-2] if 'edge' in str(n.target) else str(n.target)
        ops[k] = ops.get(k, 0) + 1
print('edge graph after lowering:', ops)
cfg = ExecutorchBackendConfig(
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True),
    segment_alignment=get_qnn_context_binary_alignment())
prog = edge.to_executorch(config=cfg)
pte = args.out + '.pte'
with open(pte, 'wb') as f: prog.write_to_file(f)
ep = prog.exported_program(); sig = ep.graph_signature
user_in = set(sig.user_inputs)
in_meta = [{'name': n.name, 'dtype': str(n.meta['val'].dtype), 'shape': list(n.meta['val'].shape)}
           for n in ep.graph.nodes if n.op == 'placeholder' and n.name in user_in]
out_node = [n for n in ep.graph.nodes if n.op == 'output'][0]
outs_meta = [{'name': a.name, 'dtype': str(a.meta['val'].dtype), 'shape': list(a.meta['val'].shape)}
             for a in out_node.args[0]]
print('pte:', pte, os.path.getsize(pte), 'bytes, build', f'{time.time()-t0:.0f}s')
print('inputs :', in_meta); print('outputs:', outs_meta)
json.dump({'pt': os.path.basename(args.pt), 'soc': args.soc, 'imgsz': args.imgsz, 'fp16': args.fp16,
           'online_prepare': args.online_prepare, 'dtype': None if args.fp16 else args.dtype, 'act_observer': None if args.fp16 else args.act_observer, 'names': names, 'calib_frames': len(frames), 'inputs': in_meta, 'outputs': outs_meta,
           'executorch': __import__('executorch.version', fromlist=['__version__']).__version__, 'order': 'per scale: box64, cls, sum'},
          open(args.out + '.json', 'w'), indent=1)
with open(args.out + '.txt', 'w') as f: f.write('\n'.join(names) + '\n')

# ---- parity: float eager / quantized eager (fake-quant) / onnx ref, decoded detections
NC, STRIDES, SIZES = len(names), (8, 16, 32), (H // 8, H // 16, H // 32)
def decode(outs, conf=0.25):
    dets = []
    for s, (stride, g_) in enumerate(zip(STRIDES, SIZES)):
        box = outs[3*s].reshape(64, g_, g_); cls = outs[3*s+1].reshape(NC, g_, g_)
        ys, xs = np.where(cls.max(0) > conf)
        for y, x in zip(ys, xs):
            b = box[:, y, x].reshape(4, 16); b = np.exp(b - b.max(1, keepdims=True)); b /= b.sum(1, keepdims=True)
            d = (b * np.arange(16)).sum(1); cx, cy = (x + 0.5) * stride, (y + 0.5) * stride
            c = int(cls[:, y, x].argmax())
            dets.append([cx - d[0]*stride, cy - d[1]*stride, cx + d[2]*stride, cy + d[3]*stride, float(cls[c, y, x]), c])
    dets = np.array(dets) if dets else np.zeros((0, 6)); keep = []
    for c in (np.unique(dets[:, 5]) if len(dets) else []):
        dc = dets[dets[:, 5] == c]; dc = dc[np.argsort(-dc[:, 4])]
        while len(dc):
            keep.append(dc[0])
            if len(dc) == 1: break
            xx1 = np.maximum(dc[0,0], dc[1:,0]); yy1 = np.maximum(dc[0,1], dc[1:,1]); xx2 = np.minimum(dc[0,2], dc[1:,2]); yy2 = np.minimum(dc[0,3], dc[1:,3])
            inter = np.clip(xx2-xx1, 0, None) * np.clip(yy2-yy1, 0, None); a0 = (dc[0,2]-dc[0,0])*(dc[0,3]-dc[0,1]); a1 = (dc[1:,2]-dc[1:,0])*(dc[1:,3]-dc[1:,1])
            dc = dc[1:][inter / (a0 + a1 - inter + 1e-9) < 0.45]
    return sorted(keep, key=lambda d: (-d[4]))
def show(tag, outs):
    ds = decode([o.astype(np.float32) for o in outs])
    print(f'  {tag:12s} ' + ' | '.join(f'{names[int(d[5])]} {d[4]:.2f} [{d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}]' for d in ds) if ds else f'  {tag:12s} (none)')
sess = None
if args.ref:
    import onnxruntime as ort
    sess = ort.InferenceSession(args.ref, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
os.makedirs(args.out + '.ref', exist_ok=True)
for i, fr in enumerate(frames[:args.check]):
    print(f'frame {i}:')
    with torch.no_grad():
        fo = [o.numpy() for o in model(fr)]
        show('float', fo)
        if not args.fp16:
            qo = [o.numpy() for o in src(fr)]; show('int8-sim', qo)
            for k, o in enumerate(qo): o.astype(np.float32).tofile(f'{args.out}.ref/f{i}_o{k}.raw')
    if sess is not None:
        show('onnx', sess.run(None, {iname: fr.numpy()}))
    fr.numpy().astype(np.float32).tofile(f'{args.out}.ref/f{i}_input.raw')
