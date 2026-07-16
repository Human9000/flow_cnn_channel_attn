#ifndef STREAM_RUNTIME_H
#define STREAM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#define SR_MAX_NODE_INPUTS 2
#define SR_MAX_TENSOR_CONSUMERS 3
#define SR_EXTERNAL_NODE UINT16_MAX

#if defined(_MSC_VER)
#define SR_ALIGN16 __declspec(align(16))
#elif defined(__GNUC__)
#define SR_ALIGN16 __attribute__((aligned(16)))
#else
#define SR_ALIGN16
#endif

typedef enum {
    SR_OP_CONV,
    SR_OP_RELU,
    SR_OP_AFFINE,
    SR_OP_AVGPOOL,
    SR_OP_UPSAMPLE,
    SR_OP_ADD,
    SR_OP_SOFTMAX,
    SR_OP_DROP
} SrOp;

typedef struct {
    uint16_t tensor;
    uint8_t reader;
} SrInput;

typedef struct {
    uint32_t offset;
    uint16_t channels;
    uint16_t capacity;
    int16_t producer;
    uint8_t consumer_count;
    uint16_t consumers[SR_MAX_TENSOR_CONSUMERS];
    uint64_t write_sequence;
    uint64_t read_sequence[SR_MAX_TENSOR_CONSUMERS];
} SrTensor;

typedef struct {
    uint16_t in_channels;
    uint16_t out_channels;
    uint16_t kernel;
    uint16_t stride;
    const float *weights;
    const float *bias;
} SrConvParams;

typedef struct {
    uint16_t channels;
    const float *scale;
    const float *bias;
} SrAffineParams;

typedef struct {
    uint16_t channels;
    uint16_t kernel;
    uint16_t stride;
} SrPoolParams;

typedef struct {
    uint16_t channels;
    uint16_t scale;
} SrUpsampleParams;

typedef struct {
    uint16_t channels;
} SrChannelParams;

typedef struct {
    SrOp op;
    uint8_t input_count;
    SrInput inputs[SR_MAX_NODE_INPUTS];
    uint16_t output;
    uint8_t queued;
    uint32_t initial_state;
    uint32_t state;
    union {
        SrConvParams conv;
        SrAffineParams affine;
        SrPoolParams pool;
        SrUpsampleParams upsample;
        SrChannelParams channel;
    } params;
} SrNode;

typedef struct {
    float *arena;
    SrTensor *tensors;
    uint16_t tensor_count;
    SrNode *nodes;
    uint16_t node_count;
    uint16_t *queue;
    uint16_t queue_head;
    uint16_t queue_tail;
    uint16_t queue_size;
} SrRuntime;

void sr_runtime_reset(SrRuntime *runtime);

int sr_runtime_push(
    SrRuntime *runtime,
    uint16_t input_tensor,
    uint16_t output_tensor,
    const float *input,
    float *output,
    uint32_t output_capacity_tokens);

#endif
