# Sampling Theorem Studio

A PyQt5-based interactive application for demonstrating and experimenting with digital signal sampling and reconstruction concepts.

## Features

- Signal Generation and Manipulation:
  - Upload custom signals from CSV files
  - Create composite signals with customizable frequency, amplitude, and phase
  - Add/remove signal components dynamically
  - Save generated signals to CSV

- Sampling and Reconstruction:
  - Adjustable sampling frequency with real-time visualization
  - Multiple reconstruction methods:
    - Whittaker-Shannon Interpolation
    - Zero-Order Hold
    - Linear Interpolation
  - Visualization of sampling points and reconstructed signal

- Analysis Tools:
  - Real-time frequency domain visualization
  - Error signal display
  - Adjustable noise addition (SNR control)
  - Frequency spectrum analysis with aliasing visualization

## Setup

1. Install required dependencies:
```bash
pip install PyQt5 numpy pyqtgraph scipy pandas
```

2. Run the application:
```bash
python final.py
```

## Usage

### Signal Creation
- Use the signal mixer to add components with different frequencies, amplitudes, and phases
- Upload existing signals using the "Upload" button
- Save generated signals using the "Save signal" button

### Signal Analysis
- Adjust sampling frequency using the slider
- Select different reconstruction methods from the dropdown menu
- Add noise to the signal using the SNR slider
- Observe frequency domain representation and aliasing effects

### Controls
- Signal Mixer: Add frequency components with custom parameters
- Sampling Frequency: Control the sampling rate relative to maximum frequency
- Reconstruction Method: Choose between different interpolation techniques
- Noise Control: Adjust Signal-to-Noise Ratio

## Interface

The application features four main display panels:
1. Original Signal with Sample Points
2. Reconstructed Signal
3. Difference (Error Signal)
4. Frequency Domain Analysis

## File Formats

- Input/Output: CSV files with two columns (Time, Amplitude)