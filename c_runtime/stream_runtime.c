#include "stream_runtime.h"

#include <float.h>
#include <math.h>
#include <string.h>

static float *sr_slot(SrRuntime *runtime, SrTensor *tensor, uint64_t sequence) {
    uint32_t slot = (uint32_t)(sequence % tensor->capacity);
    return runtime->arena + tensor->offset + slot * tensor->channels;
}

static uint64_t sr_available(const SrRuntime *runtime, SrInput input) {
    const SrTensor *tensor = &runtime->tensors[input.tensor];
    return tensor->write_sequence - tensor->read_sequence[input.reader];
}

static uint64_t sr_used(const SrTensor *tensor) {
    uint64_t minimum = tensor->read_sequence[0];
    uint8_t reader;
    for (reader = 1; reader < tensor->consumer_count; ++reader) {
        if (tensor->read_sequence[reader] < minimum) {
            minimum = tensor->read_sequence[reader];
        }
    }
    return tensor->write_sequence - minimum;
}

static uint64_t sr_free(const SrTensor *tensor) {
    return tensor->capacity - sr_used(tensor);
}

static void sr_enqueue(SrRuntime *runtime, uint16_t node_id);

static int sr_ready(const SrRuntime *runtime, uint16_t node_id) {
    const SrNode *node = &runtime->nodes[node_id];
    const SrTensor *output = &runtime->tensors[node->output];
    uint64_t free_tokens = sr_free(output);
    uint64_t first = sr_available(runtime, node->inputs[0]);

    switch (node->op) {
        case SR_OP_CONV:
            return first >= node->params.conv.kernel && free_tokens >= 1;
        case SR_OP_RELU:
        case SR_OP_AFFINE:
        case SR_OP_SOFTMAX:
            return first >= 1 && free_tokens >= 1;
        case SR_OP_AVGPOOL:
            return first >= node->params.pool.kernel && free_tokens >= 1;
        case SR_OP_UPSAMPLE:
            return first >= 1 && free_tokens >= node->params.upsample.scale;
        case SR_OP_ADD:
            return first >= 1 && sr_available(runtime, node->inputs[1]) >= 1
                   && free_tokens >= 1;
        case SR_OP_DROP:
            return first >= 1 && (node->state > 0 || free_tokens >= 1);
    }
    return 0;
}

static void sr_enqueue(SrRuntime *runtime, uint16_t node_id) {
    SrNode *node;
    if (node_id == SR_EXTERNAL_NODE) {
        return;
    }
    node = &runtime->nodes[node_id];
    if (node->queued || !sr_ready(runtime, node_id)) {
        return;
    }
    runtime->queue[runtime->queue_tail] = node_id;
    runtime->queue_tail = (uint16_t)((runtime->queue_tail + 1) % runtime->node_count);
    runtime->queue_size += 1;
    node->queued = 1;
}

static void sr_notify_write(SrRuntime *runtime, uint16_t tensor_id) {
    SrTensor *tensor = &runtime->tensors[tensor_id];
    uint8_t index;
    for (index = 0; index < tensor->consumer_count; ++index) {
        sr_enqueue(runtime, tensor->consumers[index]);
    }
}

static void sr_notify_read(SrRuntime *runtime, uint16_t tensor_id) {
    SrTensor *tensor = &runtime->tensors[tensor_id];
    if (tensor->producer >= 0) {
        sr_enqueue(runtime, (uint16_t)tensor->producer);
    }
}

static void sr_advance_input(SrRuntime *runtime, SrInput input, uint64_t count) {
    runtime->tensors[input.tensor].read_sequence[input.reader] += count;
}

static const float *sr_input_slot(
    SrRuntime *runtime, SrInput input, uint64_t relative) {
    SrTensor *tensor = &runtime->tensors[input.tensor];
    return sr_slot(runtime, tensor,
                   tensor->read_sequence[input.reader] + relative);
}

static float *sr_output_slot(SrRuntime *runtime, uint16_t tensor_id, uint64_t relative) {
    SrTensor *tensor = &runtime->tensors[tensor_id];
    return sr_slot(runtime, tensor, tensor->write_sequence + relative);
}

static void sr_run_conv(SrRuntime *runtime, SrNode *node) {
    SrConvParams *params = &node->params.conv;
    SrTensor *output = &runtime->tensors[node->output];
    uint64_t available = sr_available(runtime, node->inputs[0]);
    uint64_t count = (available - params->kernel) / params->stride + 1;
    uint64_t free_tokens = sr_free(output);
    uint64_t item;
    if (count > free_tokens) {
        count = free_tokens;
    }
    for (item = 0; item < count; ++item) {
        float *destination = sr_output_slot(runtime, node->output, item);
        uint16_t out_channel;
        for (out_channel = 0; out_channel < params->out_channels; ++out_channel) {
            float sum = params->bias[out_channel];
            uint16_t in_channel;
            for (in_channel = 0; in_channel < params->in_channels; ++in_channel) {
                uint16_t kernel_index;
                for (kernel_index = 0; kernel_index < params->kernel; ++kernel_index) {
                    const float *source = sr_input_slot(
                        runtime, node->inputs[0],
                        item * params->stride + kernel_index);
                    size_t weight_index =
                        ((size_t)out_channel * params->in_channels + in_channel)
                        * params->kernel + kernel_index;
                    sum += source[in_channel] * params->weights[weight_index];
                }
            }
            destination[out_channel] = sum;
        }
    }
    sr_advance_input(runtime, node->inputs[0], count * params->stride);
    output->write_sequence += count;
}

static void sr_run_relu(SrRuntime *runtime, SrNode *node) {
    SrTensor *output = &runtime->tensors[node->output];
    uint64_t count = sr_available(runtime, node->inputs[0]);
    uint64_t free_tokens = sr_free(output);
    uint64_t item;
    uint16_t channels = node->params.channel.channels;
    if (count > free_tokens) {
        count = free_tokens;
    }
    for (item = 0; item < count; ++item) {
        const float *source = sr_input_slot(runtime, node->inputs[0], item);
        float *destination = sr_output_slot(runtime, node->output, item);
        uint16_t channel;
        for (channel = 0; channel < channels; ++channel) {
            destination[channel] = source[channel] > 0.0f ? source[channel] : 0.0f;
        }
    }
    sr_advance_input(runtime, node->inputs[0], count);
    output->write_sequence += count;
}

static void sr_run_affine(SrRuntime *runtime, SrNode *node) {
    SrTensor *output = &runtime->tensors[node->output];
    SrAffineParams *params = &node->params.affine;
    uint64_t count = sr_available(runtime, node->inputs[0]);
    uint64_t free_tokens = sr_free(output);
    uint64_t item;
    if (count > free_tokens) {
        count = free_tokens;
    }
    for (item = 0; item < count; ++item) {
        const float *source = sr_input_slot(runtime, node->inputs[0], item);
        float *destination = sr_output_slot(runtime, node->output, item);
        uint16_t channel;
        for (channel = 0; channel < params->channels; ++channel) {
            destination[channel] = source[channel] * params->scale[channel]
                                   + params->bias[channel];
        }
    }
    sr_advance_input(runtime, node->inputs[0], count);
    output->write_sequence += count;
}

static void sr_run_pool(SrRuntime *runtime, SrNode *node) {
    SrTensor *output = &runtime->tensors[node->output];
    SrPoolParams *params = &node->params.pool;
    uint64_t available = sr_available(runtime, node->inputs[0]);
    uint64_t count = (available - params->kernel) / params->stride + 1;
    uint64_t free_tokens = sr_free(output);
    uint64_t item;
    if (count > free_tokens) {
        count = free_tokens;
    }
    for (item = 0; item < count; ++item) {
        float *destination = sr_output_slot(runtime, node->output, item);
        uint16_t channel;
        for (channel = 0; channel < params->channels; ++channel) {
            float sum = 0.0f;
            uint16_t kernel_index;
            for (kernel_index = 0; kernel_index < params->kernel; ++kernel_index) {
                const float *source = sr_input_slot(
                    runtime, node->inputs[0],
                    item * params->stride + kernel_index);
                sum += source[channel];
            }
            destination[channel] = sum / params->kernel;
        }
    }
    sr_advance_input(runtime, node->inputs[0], count * params->stride);
    output->write_sequence += count;
}

static void sr_run_upsample(SrRuntime *runtime, SrNode *node) {
    SrTensor *output = &runtime->tensors[node->output];
    SrUpsampleParams *params = &node->params.upsample;
    uint64_t count = sr_available(runtime, node->inputs[0]);
    uint64_t free_tokens = sr_free(output) / params->scale;
    uint64_t item;
    if (count > free_tokens) {
        count = free_tokens;
    }
    for (item = 0; item < count; ++item) {
        const float *source = sr_input_slot(runtime, node->inputs[0], item);
        uint16_t copy;
        for (copy = 0; copy < params->scale; ++copy) {
            float *destination = sr_output_slot(
                runtime, node->output, item * params->scale + copy);
            memcpy(destination, source, params->channels * sizeof(float));
        }
    }
    sr_advance_input(runtime, node->inputs[0], count);
    output->write_sequence += count * params->scale;
}

static void sr_run_add(SrRuntime *runtime, SrNode *node) {
    SrTensor *output = &runtime->tensors[node->output];
    uint64_t count = sr_available(runtime, node->inputs[0]);
    uint64_t second = sr_available(runtime, node->inputs[1]);
    uint64_t free_tokens = sr_free(output);
    uint16_t channels = node->params.channel.channels;
    uint64_t item;
    if (count > second) {
        count = second;
    }
    if (count > free_tokens) {
        count = free_tokens;
    }
    for (item = 0; item < count; ++item) {
        const float *left = sr_input_slot(runtime, node->inputs[0], item);
        const float *right = sr_input_slot(runtime, node->inputs[1], item);
        float *destination = sr_output_slot(runtime, node->output, item);
        uint16_t channel;
        for (channel = 0; channel < channels; ++channel) {
            destination[channel] = left[channel] + right[channel];
        }
    }
    sr_advance_input(runtime, node->inputs[0], count);
    sr_advance_input(runtime, node->inputs[1], count);
    output->write_sequence += count;
}

static void sr_run_softmax(SrRuntime *runtime, SrNode *node) {
    SrTensor *output = &runtime->tensors[node->output];
    uint64_t count = sr_available(runtime, node->inputs[0]);
    uint64_t free_tokens = sr_free(output);
    uint16_t channels = node->params.channel.channels;
    uint64_t item;
    if (count > free_tokens) {
        count = free_tokens;
    }
    for (item = 0; item < count; ++item) {
        const float *source = sr_input_slot(runtime, node->inputs[0], item);
        float *destination = sr_output_slot(runtime, node->output, item);
        float maximum = -FLT_MAX;
        float sum = 0.0f;
        uint16_t channel;
        for (channel = 0; channel < channels; ++channel) {
            if (source[channel] > maximum) {
                maximum = source[channel];
            }
        }
        for (channel = 0; channel < channels; ++channel) {
            destination[channel] = expf(source[channel] - maximum);
            sum += destination[channel];
        }
        for (channel = 0; channel < channels; ++channel) {
            destination[channel] /= sum;
        }
    }
    sr_advance_input(runtime, node->inputs[0], count);
    output->write_sequence += count;
}

static void sr_run_drop(SrRuntime *runtime, SrNode *node) {
    SrTensor *output = &runtime->tensors[node->output];
    uint64_t available = sr_available(runtime, node->inputs[0]);
    uint64_t dropped = available < node->state ? available : node->state;
    uint64_t count;
    uint64_t free_tokens;
    uint64_t item;
    node->state -= (uint32_t)dropped;
    sr_advance_input(runtime, node->inputs[0], dropped);
    available -= dropped;
    free_tokens = sr_free(output);
    count = available < free_tokens ? available : free_tokens;
    for (item = 0; item < count; ++item) {
        const float *source = sr_input_slot(runtime, node->inputs[0], item);
        float *destination = sr_output_slot(runtime, node->output, item);
        memcpy(destination, source, output->channels * sizeof(float));
    }
    sr_advance_input(runtime, node->inputs[0], count);
    output->write_sequence += count;
}

static void sr_run_node(SrRuntime *runtime, uint16_t node_id) {
    SrNode *node = &runtime->nodes[node_id];
    uint8_t input_index;
    switch (node->op) {
        case SR_OP_CONV: sr_run_conv(runtime, node); break;
        case SR_OP_RELU: sr_run_relu(runtime, node); break;
        case SR_OP_AFFINE: sr_run_affine(runtime, node); break;
        case SR_OP_AVGPOOL: sr_run_pool(runtime, node); break;
        case SR_OP_UPSAMPLE: sr_run_upsample(runtime, node); break;
        case SR_OP_ADD: sr_run_add(runtime, node); break;
        case SR_OP_SOFTMAX: sr_run_softmax(runtime, node); break;
        case SR_OP_DROP: sr_run_drop(runtime, node); break;
    }
    for (input_index = 0; input_index < node->input_count; ++input_index) {
        sr_notify_read(runtime, node->inputs[input_index].tensor);
    }
    sr_notify_write(runtime, node->output);
}

static void sr_run_queue(SrRuntime *runtime) {
    while (runtime->queue_size > 0) {
        uint16_t node_id = runtime->queue[runtime->queue_head];
        runtime->queue_head = (uint16_t)((runtime->queue_head + 1) % runtime->node_count);
        runtime->queue_size -= 1;
        runtime->nodes[node_id].queued = 0;
        if (sr_ready(runtime, node_id)) {
            sr_run_node(runtime, node_id);
        }
    }
}

void sr_runtime_reset(SrRuntime *runtime) {
    uint16_t tensor_index;
    uint16_t node_index;
    runtime->queue_head = 0;
    runtime->queue_tail = 0;
    runtime->queue_size = 0;
    for (tensor_index = 0; tensor_index < runtime->tensor_count; ++tensor_index) {
        SrTensor *tensor = &runtime->tensors[tensor_index];
        uint8_t reader;
        tensor->write_sequence = 0;
        for (reader = 0; reader < tensor->consumer_count; ++reader) {
            tensor->read_sequence[reader] = 0;
        }
    }
    for (node_index = 0; node_index < runtime->node_count; ++node_index) {
        runtime->nodes[node_index].queued = 0;
        runtime->nodes[node_index].state = runtime->nodes[node_index].initial_state;
    }
}

static int sr_write_external(
    SrRuntime *runtime, uint16_t tensor_id, const float *input) {
    SrTensor *tensor = &runtime->tensors[tensor_id];
    float *destination;
    if (sr_free(tensor) < 1) {
        return -1;
    }
    destination = sr_slot(runtime, tensor, tensor->write_sequence);
    memcpy(destination, input, tensor->channels * sizeof(float));
    tensor->write_sequence += 1;
    sr_notify_write(runtime, tensor_id);
    return 0;
}

static int sr_read_external(
    SrRuntime *runtime, uint16_t tensor_id, float *output) {
    SrTensor *tensor = &runtime->tensors[tensor_id];
    uint8_t reader = (uint8_t)(tensor->consumer_count - 1);
    const float *source;
    if (tensor->write_sequence == tensor->read_sequence[reader]) {
        return 0;
    }
    source = sr_slot(runtime, tensor, tensor->read_sequence[reader]);
    memcpy(output, source, tensor->channels * sizeof(float));
    tensor->read_sequence[reader] += 1;
    sr_notify_read(runtime, tensor_id);
    return 1;
}

int sr_runtime_push(
    SrRuntime *runtime,
    uint16_t input_tensor,
    uint16_t output_tensor,
    const float *input,
    float *output,
    uint32_t output_capacity_tokens) {
    SrTensor *result = &runtime->tensors[output_tensor];
    uint32_t output_count = 0;
    if (sr_write_external(runtime, input_tensor, input) != 0) {
        return -1;
    }
    sr_run_queue(runtime);
    while (output_count < output_capacity_tokens) {
        float *destination = output + (size_t)output_count * result->channels;
        if (!sr_read_external(runtime, output_tensor, destination)) {
            break;
        }
        output_count += 1;
        sr_run_queue(runtime);
    }
    return (int)output_count;
}

