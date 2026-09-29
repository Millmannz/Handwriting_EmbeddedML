import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DigitClassifier



def compute_scale_zero_point(x_min, x_max, q_min=0, q_max=255):
    scale = (x_max - x_min)/(q_max - q_min)
    zero_point = (q_min) - round(x_min / scale)
    # clamp zero_point to valid range, in case of edge cases
    zero_point = max(q_min, min(q_max, zero_point))
    return scale, zero_point

def quantized_linear_layer(q_x, q_w, W_sum, X_sum, zp_x, zp_w, scale_x, scale_w, N):
    # q_x: shape (N,) - quantized input for this layer
    # q_w: shape (num_neurons, N) - quantized weight matrix
    # W_sum: shape (num_neurons,) - precomputed per-neuron weight sums
    # returns: shape (num_neurons,) - float accumulator per neuron
    term1 = q_w.astype(np.int32) @ q_x.astype(np.int32)
    term2 = zp_x * W_sum
    term3 = zp_w * X_sum
    term4 = zp_x * zp_w * N
    accumulator = scale_w * scale_x * (term1 - term2 - term3 + term4)
    return accumulator

def quantized_relu(accumulator, scale_pre_relu, zp_pre_relu, scale_post_relu, zp_post_relu):
    q_pre_relu = np.round(accumulator / scale_pre_relu) + zp_pre_relu
    q_pre_relu = np.clip(q_pre_relu, 0, 255).astype(np.uint8)

    q_post_relu_raw = np.maximum(q_pre_relu, zp_pre_relu)

    float_post_relu = (q_post_relu_raw.astype(np.int32) - zp_pre_relu) * scale_pre_relu

    q_post_relu = np.round(float_post_relu / scale_post_relu) + zp_post_relu
    q_post_relu = np.clip(q_post_relu, 0, 255).astype(np.uint8)


    return q_post_relu

def array_to_c(name, array, dtype="uint8_t"):
    flat = array.flatten()
    values = ", ".join(str(int(v)) for v in flat)
    return f"static const {dtype} {name}[{flat.size}] = {{ {values} }};\n"


    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load calibration dataset
calib_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())


# 2. Build DataLoader

calib_loader = DataLoader(
    calib_data,
    batch_size=64,
    shuffle = False,
    num_workers = 0,
    pin_memory = True,
)

# 3. Load trained model
model = DigitClassifier(784, 64, 10).to(device)
model.load_state_dict(torch.load("weights.pth", map_location=device))
model.eval()


print("Layer 1 weight range:", model.network[0].weight.min(), model.network[0].weight.max())
print("Layer 2 weight range:", model.network[2].weight.min(), model.network[2].weight.max())


running_min = {"input": float("inf"), "pre_relu": float("inf"), "post_relu": float("inf"), "output": float("inf")}
running_max = {"input": float("-inf"), "pre_relu": float("-inf"), "post_relu": float("-inf"), "output": float("-inf")}

with torch.no_grad():
    for images, labels in calib_loader:
        images = images.to(device)
        x = torch.flatten(images, 1, -1)  # (a) input

        pre_relu = model.network[0](x)   # pass x through model.network[0]
        post_relu = model.network[1](pre_relu)  # pass pre_relu through model.network[1]
        output = model.network[2](post_relu)   # pass post_relu through model.network[2]

        running_min["input"] = min(running_min["input"], x.min().item())
        running_min["pre_relu"] = min(running_min["pre_relu"], pre_relu.min().item())
        running_min["post_relu"] = min(running_min["post_relu"], post_relu.min().item())
        running_min["output"] = min(running_min["output"], output.min().item())

        running_max["input"] = max(running_max["input"], x.max().item())
        running_max["pre_relu"] = max(running_max["pre_relu"], pre_relu.max().item())
        running_max["post_relu"] = max(running_max["post_relu"], post_relu.max().item())
        running_max["output"] = max(running_max["output"], output.max().item())

# --- Quantize layer 1 weights (done once, doesn't depend on input) ---
wrong_float_count = 0
wrong_quantized_count = 0
for i in range(10000):
    weight = model.network[0].weight.detach().cpu().numpy()  # shape (64, 784)
    scale_w1, zp_w1 = compute_scale_zero_point(weight.min(), weight.max())

    q_w1 = np.round(weight / scale_w1) + zp_w1
    q_w1 = np.clip(q_w1, 0, 255).astype(np.uint8)
    W_sum_1 = np.sum(q_w1, axis=1)  # shape (64,)

    # --- Quantize one input image ---
    image, label = calib_data[i]
    image_flat = image.numpy().flatten()  # shape (784,), values 0-1
    scale_x, zp_x = compute_scale_zero_point(running_min["input"], running_max["input"])

    q_x = np.round(image_flat / scale_x) + zp_x
    q_x = np.clip(q_x, 0, 255).astype(np.uint8)
    X_sum = np.sum(q_x)

    accumulator = quantized_linear_layer(q_x, q_w1, W_sum_1, X_sum, zp_x, zp_w1, scale_x, scale_w1, 784)

    scale_pre_relu, zp_pre_relu = compute_scale_zero_point(running_min["pre_relu"], running_max["pre_relu"])
    scale_post_relu, zp_post_relu = compute_scale_zero_point(running_min["post_relu"], running_max["post_relu"])

    accumulator_post_relu = quantized_relu(accumulator, scale_pre_relu, zp_pre_relu, scale_post_relu, zp_post_relu)

    weight_post_relu = model.network[2].weight.detach().cpu().numpy()
    scale_w2, zp_w2 = compute_scale_zero_point(weight_post_relu.min(), weight_post_relu.max())

    q_w2 = np.round(weight_post_relu / scale_w2) + zp_w2
    q_w2 = np.clip(q_w2, 0, 255).astype(np.uint8)

    W_sum_2 = np.sum(q_w2, axis=1)


    q_x = accumulator_post_relu
    X_sum = np.sum(q_x)

    accumulator_output = quantized_linear_layer(q_x, q_w2, W_sum_2, X_sum, zp_post_relu, zp_w2, scale_post_relu, scale_w2, 64)

    # --- Compare against the float model's layer-1 output, for validation ---
    image_tensor_flat = torch.flatten(image.unsqueeze(0).to(device), 1, -1)
    with torch.no_grad():
        float_output = model.network[0](image_tensor_flat)
        float_output = model.network[1](float_output)
        float_output = model.network[2](float_output)
    float_output_np = float_output[0].cpu().numpy()

    diff = accumulator_output - float_output_np

    # Check accuracy
    if np.argmax(accumulator_output) != label:
        wrong_quantized_count+=1
    if np.argmax(float_output_np) != label:
        wrong_float_count+=1
    if np.argmax(accumulator_output) != np.argmax(float_output_np):
        print(f"Different prediction: {i}")

print(f"Total float wrong count: {wrong_float_count} - Acc: {(10000-wrong_float_count)/10000}")
print(f"Total quantized wrong count: {wrong_quantized_count} - Acc: {(10000-wrong_quantized_count)/10000}")
print(f"Accuracy Loss From quantization: {(wrong_quantized_count-wrong_float_count)/10000}")


def generate_header(filepath):
    lines = []
    lines.append("#ifndef MODEL_WEIGHTS_H")
    lines.append("#define MODEL_WEIGHTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")

    # Layer 1
    lines.append(array_to_c("layer1_weights", q_w1))
    lines.append(array_to_c("layer1_W_sum", W_sum_1, dtype="int32_t"))
    lines.append(f"#define LAYER1_SCALE_W {scale_w1}f")
    lines.append(f"#define LAYER1_ZP_W {zp_w1}")
    lines.append("")

    # Input quantization
    lines.append(f"#define INPUT_SCALE {scale_x}f")
    lines.append(f"#define INPUT_ZP {zp_x}")
    lines.append("")

    # Pre/post ReLU quantization
    lines.append(f"#define PRE_RELU_SCALE {scale_pre_relu}f")
    lines.append(f"#define PRE_RELU_ZP {zp_pre_relu}")
    lines.append(f"#define POST_RELU_SCALE {scale_post_relu}f")
    lines.append(f"#define POST_RELU_ZP {zp_post_relu}")
    lines.append("")

    # Layer 2
    lines.append(array_to_c("layer2_weights", q_w2))
    lines.append(array_to_c("layer2_W_sum", W_sum_2, dtype="int32_t"))
    lines.append(f"#define LAYER2_SCALE_W {scale_w2}f")
    lines.append(f"#define LAYER2_ZP_W {zp_w2}")
    lines.append("")

    lines.append("#define INPUT_SIZE 784")
    lines.append("#define HIDDEN_SIZE 64")
    lines.append("#define OUTPUT_SIZE 10")
    lines.append("")
    lines.append("#endif")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

generate_header("HandwritingML/Core/Inc/model_weights.h")