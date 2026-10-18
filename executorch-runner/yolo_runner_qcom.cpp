/*
 * The yolo runner (lib/model/yolo_runner.h) for Qualcomm boards: the
 * dispatcher over the QNN context-binary runner and, when built with
 * ExecuTorch, the .pte runner (qcom_runner.h). Per frame: a BGR frame is
 * swapped to RGB, letterboxed on the CPU into the model's input size,
 * handed to the runner, and the outputs go to the backend-neutral
 * post-processor. One inference slot: the NPU is driven from one thread.
 *
 * Dump: with EDGEBLE_FEAT_DUMP=<dir> every EDGEBLE_FEAT_DUMP_EVERY-th
 * frame (default 10) is written as <dir>/<n>.rgb, <dir>/<n>.feats and
 * <dir>/<n>.det.
 */
#include "model/yolo_runner.h"
#include "model/yolo11/yolo11_postprocess.h"
#include "qcom_runner.h"
#include "utils/edgeble_perf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <mutex>
#include <vector>

struct edgeble_yolo_ctx {
	const qcom_runner *runner = nullptr;
	void *impl = nullptr;
	int model_w = 0, model_h = 0;
	std::vector<uint8_t> letterboxed;	/* RGB888 HWC, model_w x model_h */
	std::vector<uint8_t> rgb_copy;		/* when the frame comes as BGR888 */
	std::mutex lock;
	std::string dump_dir;
	int dump_every = 10;
	long frame = 0;
};

static size_t tensor_numel(const ed_tensor_t &t)
{
	size_t n = 1;
	for (int d = 0; d < t.n_dims; d++)
		n *= t.dims[d];
	return n;
}

static void dump_frame(edgeble_yolo_ctx *ctx, const object_detect_result_list &results)
{
	const ed_tensor_t *feats = nullptr;
	int n_feats = 0;
	char base[512];

	if (!ctx->runner->aux || ctx->runner->aux(ctx->impl, &feats, &n_feats) < 0 || n_feats < 1)
		return;
	snprintf(base, sizeof(base), "%s/%06ld", ctx->dump_dir.c_str(), ctx->frame);
	std::string p = std::string(base) + ".rgb";
	FILE *f = fopen(p.c_str(), "wb");
	if (!f) {
		printf("[qcom] cannot write %s, dump off\n", p.c_str());
		ctx->dump_dir.clear();
		return;
	}
	fwrite(ctx->letterboxed.data(), 1, ctx->letterboxed.size(), f);
	fclose(f);
	p = std::string(base) + ".feats";
	f = fopen(p.c_str(), "wb");
	if (f) {
		for (int i = 0; i < n_feats; i++)
			if (feats[i].type == ED_TENSOR_F32)
				fwrite(feats[i].data, sizeof(float), tensor_numel(feats[i]), f);
		fclose(f);
	}
	p = std::string(base) + ".det";
	f = fopen(p.c_str(), "w");
	if (f) {
		for (int i = 0; i < results.count; i++) {
			const object_detect_result *r = &results.results[i];
			fprintf(f, "%d %.4f %d %d %d %d\n", r->cls_id, r->prop, r->box.left, r->box.top,
				r->box.right, r->box.bottom);
		}
		fclose(f);
	}
	if (ctx->frame == 0) {
		p = ctx->dump_dir + "/feats.json";
		f = fopen(p.c_str(), "w");
		if (f) {
			fprintf(f, "{\"model_w\": %d, \"model_h\": %d, \"feats\": [", ctx->model_w, ctx->model_h);
			for (int i = 0; i < n_feats; i++) {
				fprintf(f, "%s[", i ? ", " : "");
				for (int d = 0; d < feats[i].n_dims; d++)
					fprintf(f, "%s%u", d ? ", " : "", feats[i].dims[d]);
				fprintf(f, "]");
			}
			fprintf(f, "], \"det\": \"class conf x1 y1 x2 y2 (letterbox px)\"}\n");
			fclose(f);
		}
	}
}

std::string qcom_dir_of(const char *path)
{
	std::string s(path);
	size_t p = s.find_last_of('/');
	return p == std::string::npos ? "." : s.substr(0, p);
}

static bool file_exists(const std::string &p)
{
	struct stat st;
	return stat(p.c_str(), &st) == 0;
}

std::string qcom_find_lib(const char *model_path, const char *name)
{
	const char *env = getenv("EDGEBLE_QNN_LIBS");
	std::vector<std::string> dirs;

	if (env && *env)
		dirs.push_back(env);
	dirs.push_back(qcom_dir_of(model_path));
	dirs.push_back("/usr/lib");
	dirs.push_back("/usr/lib/aarch64-linux-gnu");
	dirs.push_back("/usr/local/lib");
	for (const std::string &d : dirs)
		if (file_exists(d + "/" + name))
			return d + "/" + name;
	return name;	/* let the dynamic loader search */
}

void qcom_set_adsp_path(const std::string &libdir)
{
	if (!getenv("ADSP_LIBRARY_PATH")) {
		std::string p = libdir + ";/usr/lib/rfsa/adsp;/dsp";
		setenv("ADSP_LIBRARY_PATH", p.c_str(), 1);
	}
}

int edgeble_yolo_init_labels(void)
{
	return edgeble_yolo11_init_post_process();
}

void edgeble_yolo_deinit_labels(void)
{
	edgeble_yolo11_deinit_post_process();
}

void edgeble_yolo_set_class_num(int class_num)
{
	edgeble_yolo11_set_post_process_class_num(class_num);
}

const char *edgeble_yolo_class_name(int cls_id)
{
	return edgeble_yolo11_cls_to_name(cls_id);
}

static bool has_suffix(const char *s, const char *suffix)
{
	size_t n = strlen(s), m = strlen(suffix);
	return n >= m && strcasecmp(s + n - m, suffix) == 0;
}

int edgeble_yolo_init(const char *model_path, edgeble_yolo_ctx **ctx)
{
	if (!model_path || !ctx)
		return -1;
	*ctx = nullptr;

	const qcom_runner *runner = &qcom_runner_qnn;
	if (has_suffix(model_path, ".pte")) {
#if EDGEBLE_HAS_EXECUTORCH
		runner = &qcom_runner_executorch;
#else
		printf("[qcom] %s is an ExecuTorch program but this build has no ExecuTorch runtime "
		       "(build with -e <executorch install>)\n", model_path);
		return -1;
#endif
	}

	edgeble_yolo_ctx *c = new edgeble_yolo_ctx();
	c->runner = runner;
	if (runner->init(model_path, &c->impl, &c->model_w, &c->model_h) < 0 || !c->impl ||
	    c->model_w <= 0 || c->model_h <= 0) {
		if (c->impl)
			runner->release(c->impl);
		delete c;
		return -1;
	}
	c->letterboxed.assign((size_t)c->model_w * c->model_h * 3, 114);
	if (const char *d = getenv("EDGEBLE_FEAT_DUMP")) {
		if (*d && runner->aux) {
			c->dump_dir = d;
			if (const char *e = getenv("EDGEBLE_FEAT_DUMP_EVERY"))
				c->dump_every = atoi(e) > 0 ? atoi(e) : 1;
			printf("[qcom] training dump every %d frames into %s\n", c->dump_every, d);
		} else if (*d) {
			printf("[qcom] EDGEBLE_FEAT_DUMP ignored: %s has no float tail\n", runner->name);
		}
	}
	printf("model input height=%d, width=%d, channel=3 (%s)\n", c->model_h, c->model_w, runner->name);
	printf("[npu] inference slots: 1\n");
	*ctx = c;
	return 0;
}

int edgeble_yolo_slots(edgeble_yolo_ctx *ctx)
{
	return ctx ? 1 : 0;
}

int edgeble_yolo_infer_slot(edgeble_yolo_ctx *ctx, int slot,
			    image_buffer_t *img, ed_object_t *objects,
			    int max_objects, int *n_objects)
{
	if (!ctx || !img || !objects || !n_objects || slot != 0)
		return -1;
	*n_objects = 0;

	std::lock_guard<std::mutex> guard(ctx->lock);
	image_buffer_t src = *img;
	image_buffer_t dst;
	letterbox_t box;
	object_detect_result_list results;
	const ed_tensor_t *outs = nullptr;
	int n_outs = 0;

	/* the CPU letterbox takes RGB888; a BGR frame (OpenCV) is swapped first */
	if (src.format == IMAGE_FORMAT_BGR888) {
		size_t n = (size_t)src.width * src.height * 3;
		ctx->rgb_copy.resize(n);
		const uint8_t *b = src.virt_addr;
		uint8_t *r = ctx->rgb_copy.data();
		for (size_t i = 0; i < n; i += 3) {
			r[i] = b[i + 2]; r[i + 1] = b[i + 1]; r[i + 2] = b[i];
		}
		src.virt_addr = r;
		src.format = IMAGE_FORMAT_RGB888;
		src.fd = -1;
	}
	memset(&dst, 0, sizeof(dst));
	memset(&box, 0, sizeof(box));
	dst.width = ctx->model_w;
	dst.height = ctx->model_h;
	dst.format = IMAGE_FORMAT_RGB888;
	dst.size = get_image_size(&dst);
	dst.virt_addr = ctx->letterboxed.data();
	dst.fd = -1;
	{
		ED_PERF_SCOPE("infer.letterbox");
		if (convert_image_with_letterbox(&src, &dst, &box, 114) < 0) {
			printf("[qcom] letterbox failed\n");
			return -1;
		}
	}
	if (ctx->runner->run(ctx->impl, ctx->letterboxed.data(), &outs, &n_outs) < 0 || !outs || n_outs < 1)
		return -1;
	{
		ED_PERF_SCOPE("infer.post");
		memset(&results, 0, sizeof(results));
		edgeble_yolo11_post_process(outs, n_outs, ctx->model_w, ctx->model_h,
					    &box, BOX_THRESH, NMS_THRESH, &results);
	}
	if (!ctx->dump_dir.empty()) {
		if (ctx->frame % ctx->dump_every == 0)
			dump_frame(ctx, results);
		ctx->frame++;
	}

	int count = results.count > max_objects ? max_objects : results.count;
	for (int i = 0; i < count; i++) {
		object_detect_result *r = &results.results[i];
		ed_object_t *o = &objects[i];

		o->x = (float)r->box.left;
		o->y = (float)r->box.top;
		o->w = (float)(r->box.right - r->box.left);
		o->h = (float)(r->box.bottom - r->box.top);
		o->label = r->cls_id;
		o->prob = r->prop;
		o->has_mask = r->has_mask;
		o->mask_w = r->mask_w;
		o->mask_h = r->mask_h;
		o->mask_model_w = r->mask_model_w;
		o->mask_model_h = r->mask_model_h;
		o->mask_x_pad = r->mask_x_pad;
		o->mask_y_pad = r->mask_y_pad;
		o->mask_scale = r->mask_scale;
		o->mask = r->mask;
	}
	*n_objects = count;
	return 0;
}

void edgeble_yolo_release(edgeble_yolo_ctx *ctx)
{
	if (!ctx)
		return;
	if (ctx->impl)
		ctx->runner->release(ctx->impl);
	delete ctx;
}
