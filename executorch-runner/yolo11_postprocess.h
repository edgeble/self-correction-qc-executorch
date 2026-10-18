#ifndef EDGEBLE_YOLO11_POSTPROCESS_H
#define EDGEBLE_YOLO11_POSTPROCESS_H

#include <stdint.h>
#include <vector>
#include "common.h"
#include "image_utils.h"

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define OBJ_MASK_DIM 32
#define OBJ_MASK_MAX_W 160
#define OBJ_MASK_MAX_H 160
#define OBJ_MASK_THRESH 0.5f
#define EDGEBLE_HAS_SEGMENT_MASK 1

/* One model output as any backend hands it over: the buffer, its shape
 * as reported (the YOLO heads are 1 x C x H x W), and the quantisation
 * of int8/uint8 data (value = (q - zp) * scale). The post-processor is
 * backend-neutral; the RKNN and QNN runners fill this from their own
 * attributes. */
typedef enum {
    ED_TENSOR_F32 = 0,
    ED_TENSOR_I8,
    ED_TENSOR_U8,
} ed_tensor_type_t;

typedef struct {
    const void *data;
    uint32_t dims[4];
    int n_dims;
    ed_tensor_type_t type;
    int32_t zp;
    float scale;
} ed_tensor_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
    int has_mask;
    int mask_w;
    int mask_h;
    int mask_model_w;
    int mask_model_h;
    int mask_x_pad;
    int mask_y_pad;
    float mask_scale;
    const uint8_t *mask;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int edgeble_yolo11_init_post_process();
void edgeble_yolo11_deinit_post_process();
void edgeble_yolo11_set_post_process_class_num(int class_num);
char *edgeble_yolo11_cls_to_name(int cls_id);
/* outs: the model's outputs in order, three per scale (box 64, class,
 * score sum) plus the mask coefficients and the prototype tensor for
 * segmentation models; model_w/h: the model input size */
int edgeble_yolo11_post_process(const ed_tensor_t *outs, int n_outs, int model_w, int model_h,
                                letterbox_t *letter_box, float conf_threshold, float nms_threshold,
                                object_detect_result_list *od_results);

#endif //EDGEBLE_YOLO11_POSTPROCESS_H
