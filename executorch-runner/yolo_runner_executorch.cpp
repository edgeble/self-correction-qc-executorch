/*
 * The ExecuTorch runner behind the Qualcomm yolo runner
 * (yolo_runner_qcom.cpp): the model is a .pte, an ExecuTorch program whose
 * graph was lowered to the QNN HTP backend at export
 * (platform/qcom/README.md). The ExecuTorch runtime loads the program and
 * its registered qnn_executorch_backend runs the delegated graph on the
 * Hexagon NPU through libQnnHtp.so; whatever the export left on the CPU
 * (nothing for the nine-output YOLO head, a float tail later) runs in the
 * portable kernels of the same binary.
 *
 * The backend dlopens the QNN libraries by bare name. This runner loads
 * libQnnSystem.so and libQnnHtp.so from qcom_find_lib() first, RTLD_GLOBAL,
 * so the backend's lookup lands on the same copies without LD_LIBRARY_PATH,
 * and points ADSP_LIBRARY_PATH at that directory for the skel library.
 *
 * Input: one 4-D tensor, float32 NCHW/NHWC 0..1 (the export's x/255 is
 * done here) or uint8. Outputs: whatever the program returns, in order,
 * described as ed_tensor_t; the export dequantizes them to float32.
 *
 * The pair: `<name>-A.pte` given as the model with `<name>-B.pte` next to
 * it and the head's weights in `<name>-B.ptd` (loaded at init).
 */
#include "qcom_runner.h"
#include "utils/edgeble_perf.h"

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <mutex>
#include <vector>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {

const char *type_name(ScalarType t);

struct et_ctx {
	void *lib_system = nullptr;
	void *lib_backend = nullptr;
	std::unique_ptr<Module> module;
	int model_w = 0, model_h = 0;
	bool input_nchw = true;
	ScalarType in_type = ScalarType::Float;
	std::vector<int32_t> in_sizes;
	std::vector<float> in_f32;
	std::vector<uint8_t> in_u8;
	TensorPtr input;
	std::vector<EValue> results;		/* the last forward(), owns nothing: the method's buffers */
	std::vector<ed_tensor_t> outs;
	/* the float tail (sealed + float pair) */
	std::unique_ptr<Module> tail;
	std::vector<EValue> tail_results;
	std::vector<ed_tensor_t> feats;		/* B's inputs = A's odd outputs, the last run */
	std::vector<ed_tensor_t> head_outs;	/* [box, cls, sum] x scales */
};

bool file_exists(const std::string &p)
{
	FILE *f = fopen(p.c_str(), "rb");
	if (!f)
		return false;
	fclose(f);
	return true;
}

/* <dir>/<name>-A.pte -> <dir>/<name>-B.pte / .ptd, "" when no pair */
std::string pair_path(const std::string &model_path, const char *ext)
{
	const std::string suffix = "-A.pte";
	if (model_path.size() <= suffix.size() ||
	    model_path.compare(model_path.size() - suffix.size(), suffix.size(), suffix) != 0)
		return "";
	return model_path.substr(0, model_path.size() - suffix.size()) + "-B" + ext;
}

/* an EValue tensor described as ed_tensor_t (float32 only) */
bool describe(const EValue &v, ed_tensor_t &o, const char *what, size_t idx)
{
	if (!v.isTensor())
		return false;
	const auto &t = v.toTensor();
	int nd = (int)t.dim();

	o.data = t.const_data_ptr();
	o.n_dims = nd > 4 ? 4 : nd;
	for (int d = 0; d < 4; d++)
		o.dims[d] = d < nd ? (uint32_t)t.size(d) : 1;
	o.scale = 1.0f;
	o.zp = 0;
	if (t.scalar_type() != ScalarType::Float) {
		static bool said;
		if (!said) {
			printf("[executorch] %s output %zu is %s: only float32 outputs are handled, "
			       "export with the dequantize kept in the graph\n", what, idx, type_name(t.scalar_type()));
			said = true;
		}
		return false;
	}
	o.type = ED_TENSOR_F32;
	return true;
}

const char *type_name(ScalarType t)
{
	switch (t) {
	case ScalarType::Float: return "float32";
	case ScalarType::Half: return "float16";
	case ScalarType::Byte: return "uint8";
	case ScalarType::Char: return "int8";
	case ScalarType::Int: return "int32";
	case ScalarType::Long: return "int64";
	default: return "other";
	}
}

void teardown(et_ctx *c)
{
	c->input.reset();
	c->results.clear();
	c->tail_results.clear();
	c->tail.reset();
	c->module.reset();
	if (c->lib_backend)
		dlclose(c->lib_backend);
	if (c->lib_system)
		dlclose(c->lib_system);
	c->lib_backend = c->lib_system = nullptr;
}

int et_init(const char *model_path, void **impl, int *model_w, int *model_h)
{
	static std::once_flag once;

	if (!model_path || !impl)
		return -1;
	*impl = nullptr;
	std::call_once(once, [] { executorch::runtime::runtime_init(); });

	et_ctx *c = new et_ctx();
	std::string sys_lib = qcom_find_lib(model_path, "libQnnSystem.so");
	std::string htp_lib = qcom_find_lib(model_path, "libQnnHtp.so");
	std::string libdir = qcom_dir_of(htp_lib.c_str());

	qcom_set_adsp_path(libdir);
	c->lib_system = dlopen(sys_lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
	c->lib_backend = dlopen(htp_lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
	if (!c->lib_system || !c->lib_backend)
		printf("[executorch] QNN libraries not preloaded (%s): the backend searches the loader paths\n",
		       dlerror());

	c->module.reset(new Module(model_path, Module::LoadMode::MmapUseMlock));
	Error err = c->module->load_method("forward");
	if (err != Error::Ok) {
		printf("[executorch] cannot load %s (forward): error 0x%x\n", model_path, (unsigned)err);
		teardown(c); delete c;
		return -1;
	}
	auto meta = c->module->method_meta("forward");
	if (!meta.ok()) {
		printf("[executorch] no method meta for forward: error 0x%x\n", (unsigned)meta.error());
		teardown(c); delete c;
		return -1;
	}
	if (meta->num_inputs() != 1) {
		printf("[executorch] expected one input, got %zu\n", meta->num_inputs());
		teardown(c); delete c;
		return -1;
	}
	auto in = meta->input_tensor_meta(0);
	if (!in.ok() || in->sizes().size() != 4) {
		printf("[executorch] input 0 is not a 4-D tensor\n");
		teardown(c); delete c;
		return -1;
	}
	c->in_sizes.assign(in->sizes().begin(), in->sizes().end());
	c->in_type = in->scalar_type();
	{
		const std::vector<int32_t> &d = c->in_sizes;
		if (d[1] == 3) {
			c->input_nchw = true; c->model_h = d[2]; c->model_w = d[3];
		} else if (d[3] == 3) {
			c->input_nchw = false; c->model_h = d[1]; c->model_w = d[2];
		} else {
			printf("[executorch] input [%d,%d,%d,%d] has no 3-channel axis\n", d[0], d[1], d[2], d[3]);
			teardown(c); delete c;
			return -1;
		}
	}
	size_t numel = (size_t)c->model_w * c->model_h * 3;
	void *data = nullptr;
	switch (c->in_type) {
	case ScalarType::Float: c->in_f32.assign(numel, 0.0f); data = c->in_f32.data(); break;
	case ScalarType::Byte:  c->in_u8.assign(numel, 0); data = c->in_u8.data(); break;
	default:
		printf("[executorch] unsupported input type %s\n", type_name(c->in_type));
		teardown(c); delete c;
		return -1;
	}
	c->input = executorch::extension::make_tensor_ptr(c->in_sizes, data, c->in_type);

	printf("[executorch] %s: forward, %zu outputs, libs %s\n", model_path, meta->num_outputs(), libdir.c_str());
	printf("[executorch] input %s [%d,%d,%d,%d] (%s)\n", type_name(c->in_type), c->in_sizes[0], c->in_sizes[1],
	       c->in_sizes[2], c->in_sizes[3], c->input_nchw ? "NCHW" : "NHWC");
	for (size_t i = 0; i < meta->num_outputs(); i++) {
		auto o = meta->output_tensor_meta(i);
		if (!o.ok())
			continue;
		printf("[executorch] output %zu %s [", i, type_name(o->scalar_type()));
		for (size_t d = 0; d < o->sizes().size(); d++)
			printf("%s%d", d ? "," : "", (int)o->sizes()[d]);
		printf("]\n");
	}
	c->outs.resize(meta->num_outputs());

	std::string b_pte = pair_path(model_path, ".pte");
	if (!b_pte.empty() && file_exists(b_pte)) {
		std::string b_ptd = pair_path(model_path, ".ptd");
		if (file_exists(b_ptd))
			c->tail.reset(new Module(b_pte, b_ptd, Module::LoadMode::MmapUseMlock));
		else
			c->tail.reset(new Module(b_pte, Module::LoadMode::MmapUseMlock));
		err = c->tail->load_method("forward");
		if (err != Error::Ok) {
			printf("[executorch] cannot load the tail %s (forward): error 0x%x\n", b_pte.c_str(), (unsigned)err);
			teardown(c); delete c;
			return -1;
		}
		auto tm = c->tail->method_meta("forward");
		if (!tm.ok() || tm->num_inputs() * 2 != meta->num_outputs()) {
			printf("[executorch] tail %s takes %zu inputs, A gives %zu outputs: expected [box, feat] per scale\n",
			       b_pte.c_str(), tm.ok() ? tm->num_inputs() : 0, meta->num_outputs());
			teardown(c); delete c;
			return -1;
		}
		c->feats.resize(tm->num_inputs());
		c->head_outs.resize(tm->num_inputs() * 3);
		printf("[executorch] tail %s (%s): %zu feats in, %zu outputs\n", b_pte.c_str(),
		       file_exists(b_ptd) ? "weights from -B.ptd" : "weights inside, no -B.ptd", tm->num_inputs(), tm->num_outputs());
	}
	*impl = c;
	*model_w = c->model_w;
	*model_h = c->model_h;
	return 0;
}

/* letterboxed RGB888 HWC -> the input tensor (layout and type) */
void pack_input(et_ctx *c, const uint8_t *src)
{
	const size_t hw = (size_t)c->model_w * c->model_h;

	if (c->in_type == ScalarType::Float) {
		float *p = c->in_f32.data();
		if (c->input_nchw) {
			for (int ch = 0; ch < 3; ch++)
				for (size_t i = 0; i < hw; i++)
					p[ch * hw + i] = src[i * 3 + ch] * (1.0f / 255.0f);
		} else {
			for (size_t i = 0; i < hw * 3; i++)
				p[i] = src[i] * (1.0f / 255.0f);
		}
	} else {
		uint8_t *p = c->in_u8.data();
		if (c->input_nchw) {
			for (int ch = 0; ch < 3; ch++)
				for (size_t i = 0; i < hw; i++)
					p[ch * hw + i] = src[i * 3 + ch];
		} else {
			memcpy(p, src, hw * 3);
		}
	}
}

int et_run(void *impl, const uint8_t *rgb, const ed_tensor_t **outs, int *n_outs)
{
	et_ctx *c = (et_ctx *)impl;

	{
		ED_PERF_SCOPE("infer.pack");
		pack_input(c, rgb);
	}
	{
		ED_PERF_SCOPE("infer.executorch_run");
		auto r = c->module->forward(c->input);
		if (!r.ok()) {
			printf("[executorch] forward failed: error 0x%x\n", (unsigned)r.error());
			return -1;
		}
		c->results = std::move(*r);
	}
	size_t n = 0;
	for (const EValue &v : c->results) {
		if (n >= c->outs.size())
			break;
		if (!describe(v, c->outs[n], "A", n))
			return -1;
		n++;
	}
	if (!c->tail) {
		*outs = c->outs.data();
		*n_outs = (int)n;
		return 0;
	}

	/* the float tail: feats = A's odd outputs; B returns [cls, sum] per scale */
	size_t nl = c->feats.size();
	if (n != nl * 2)
		return -1;
	std::vector<EValue> in;
	std::vector<TensorPtr> keep;
	for (size_t i = 0; i < nl; i++) {
		const auto &t = c->results[2 * i + 1].toTensor();
		c->feats[i] = c->outs[2 * i + 1];
		std::vector<int32_t> sizes(t.sizes().begin(), t.sizes().end());
		keep.push_back(executorch::extension::make_tensor_ptr(sizes, const_cast<void *>(t.const_data_ptr()),
								      ScalarType::Float));
		in.push_back(*keep.back());
	}
	{
		ED_PERF_SCOPE("infer.tail_run");
		auto r = c->tail->forward(in);
		if (!r.ok()) {
			printf("[executorch] tail forward failed: error 0x%x\n", (unsigned)r.error());
			return -1;
		}
		c->tail_results = std::move(*r);
	}
	if (c->tail_results.size() != nl * 2)
		return -1;
	for (size_t i = 0; i < nl; i++) {
		c->head_outs[3 * i] = c->outs[2 * i];			/* box */
		if (!describe(c->tail_results[2 * i], c->head_outs[3 * i + 1], "B", 2 * i) ||
		    !describe(c->tail_results[2 * i + 1], c->head_outs[3 * i + 2], "B", 2 * i + 1))
			return -1;
	}
	*outs = c->head_outs.data();
	*n_outs = (int)(nl * 3);
	return 0;
}

int et_aux(void *impl, const ed_tensor_t **feats, int *n_feats)
{
	et_ctx *c = (et_ctx *)impl;

	if (!c->tail)
		return -1;
	*feats = c->feats.data();
	*n_feats = (int)c->feats.size();
	return 0;
}

void et_release(void *impl)
{
	et_ctx *c = (et_ctx *)impl;

	if (!c)
		return;
	teardown(c);
	delete c;
}

} /* namespace */

const qcom_runner qcom_runner_executorch = { "executorch", et_init, et_run, et_release, et_aux };
