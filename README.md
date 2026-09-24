# Handwritten Digit Recognizer — STM32G474 + Touchscreen

An embedded machine learning project that recognizes handwritten digits (0–9) drawn directly on a touchscreen, running entirely on an STM32G474 microcontroller.

## Overview

This project combines a resistive/capacitive touchscreen input with an on-device neural network to classify handwritten digits in real time. A digit is drawn on the screen, the drawing is captured and preprocessed into a format matching the MNIST dataset (28x28 grayscale), and a quantized CNN — deployed on-device — runs inference and displays the predicted digit back to the user.

## Features

- Real-time digit drawing and capture via touchscreen
- On-device preprocessing (downsampling, normalization, centering)
- Neural network inference running fully on-chip (no external compute)
- Live prediction display with confidence score
- Clear/reset functionality for repeated testing

## Hardware

| Component | Details |
|---|---|
| MCU | STM32G474 (Cortex-M4, FPU, up to 170 MHz) |
| Display/Touch | [TouchScreen(TODO:Fill in model)] |
| Interface | [I2C] |
| Power | [USB] |

## Software & Tools

- **STM32CubeIDE** — firmware development
- **STM32CubeMX** — peripheral/clock configuration
- **Python (PyTorch)** — model training on MNIST
- **STM32 HAL** — touchscreen driver and display control

## Machine Learning Pipeline

1. **Training (offline, on PC)**
   - Model trained on the MNIST handwritten digit dataset
   - Architecture: [e.g., small CNN — Conv2D → MaxPool → Conv2D → MaxPool → Dense]

2. **Conversion**
   - Model quantized and converted to optimized C code
   - Generated inference library integrated into the STM32CubeIDE project

3. **On-device Inference**
   - Touchscreen input captured and downsampled to 28x28
   - Pixel data normalized to match training preprocessing
   - Inference run on the STM32G474; output is a probability distribution over digits 0–9
   - Predicted digit + confidence shown on display

## Project Structure

```
.
├── Core/                   # STM32 firmware source (main.c, drivers, etc.)
├── model_training/         # Python scripts/notebooks for training the CNN
│   ├── train.py
│   └── export_model.py
├── Drivers/                # HAL and touchscreen/display drivers
└── README.md
```

## Getting Started

### Prerequisites
- ST-Link programmer/debugger
- Python 3.x with TensorFlow/Keras (for retraining the model, optional)

### Build & Flash
1. Clone this repository
2. Open the project in STM32CubeIDE
3. Build the project (`Project > Build`)
4. Connect the STM32G474 board via ST-Link
5. Flash the firmware (`Run > Debug` or `Run > Run`)

### Usage
1. Power on the board
2. Draw a digit (0–9) on the touchscreen
3. The predicted digit and confidence score appear on the display
4. Tap "Clear" (or equivalent) to reset and draw again

## Model Performance

| Metric | Value |
|---|---|
| Training accuracy | [Coming Soon]% |
| Test accuracy | [Coming Soon]% |
| On-device inference time | [Coming Soon] ms |
| Model size (post-quantization) | [Coming soon] KB |

## Future Improvements

- Support for full alphanumeric character recognition
- On-device retraining / personalization
- Improved touch smoothing and stroke capture
- Power optimization for battery operation

## Acknowledgments

- MNIST dataset (Yann LeCun et al.)
- STMicroelectronics toolchain
