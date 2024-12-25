import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel,
                             QHBoxLayout, QGridLayout, QPushButton, QLineEdit, QSlider,
                             QComboBox, QFrame)  # Import QFrame
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pyqtgraph as pg
from scipy.fft import fft, fftfreq
import pandas as pd
import csv


class SamplingTheorem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sampling Theorem studio")
        self.setGeometry(200, 200, 1500, 1200)
        self.components = []
        self.generated_signal = None
        self.composer_frequencies = []
        self.max_freq = 0
        self.original_fs = 1000  # Default sampling rate
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_plots)
        self.initUI()

    def initUI(self):
        main_window = QWidget()
        container = QGridLayout()
        container.setSpacing(20) #spacing between widgets
        main_window.setLayout(container)
        self.setCentralWidget(main_window)

        # Initialize four graphs using the create_graph_canvas function
        self.canvas1 = self.create_graph_canvas("Original Signal with Sample Points")
        self.canvas2 = self.create_graph_canvas("Reconstructed Signal")
        self.canvas3 = self.create_error_canvas("Difference")
        self.canvas4 = self.create_graph_canvas("Frequency Domain")

        # GRAPHS
        container.addWidget(self.canvas1, 0, 1)
        container.addWidget(self.canvas2, 0, 2)
        container.addWidget(self.canvas3, 1, 1)
        container.addWidget(self.canvas4, 1, 2)

        # CONTROLS
        controls_layout1 = QVBoxLayout()
        controls_layout2 = QVBoxLayout()
        controls_layout1.setSpacing(10)
        controls_layout2.setSpacing(10)

        self.upload_signal_label = QLabel("Upload signal:")
        self.upload_signal_button = QPushButton("Upload")
        self.upload_signal_button.clicked.connect(self.upload_and_plot_signal)

        # Reconstruction menu
        reconstruction_method_layout = QHBoxLayout()
        self.reconstruction_label = QLabel("Reconstruction method:")
        self.reconstruction_menu = QComboBox()
        self.reconstruction_menu.addItems(["Whittaker-Shannon", "Zero-Order Hold", "Linear Interpolation"])
        self.reconstruction_menu.currentIndexChanged.connect(self.queue_plot_update)
        reconstruction_method_layout.addWidget(self.reconstruction_label)
        reconstruction_method_layout.addWidget(self.reconstruction_menu)

        # Signal mixer
        self.signal_mixer_label = QLabel("Signal mixer")
        self.freq_signal_label = QLabel("Frequency")
        freq_signal_layout = QHBoxLayout()
        self.freq_slider = QSlider(Qt.Horizontal)
        self.freq_slider.setRange(0, 1000) # set fixed range for the mixer frequency slider
        self.freq_slider.setValue(0)
        self.frequency_value = QLabel("0 HZ")
        self.freq_slider.valueChanged.connect(self.update_frequency_label)
        freq_signal_layout.addWidget(self.freq_slider)
        freq_signal_layout.addWidget(self.frequency_value)

        self.amplitude_signal_label = QLabel("Amplitude")
        amplitude_signal_layout = QHBoxLayout()
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setRange(0, 100)
        self.amplitude_slider.setValue(0)
        self.amplitude_value = QLabel("0")
        self.amplitude_slider.valueChanged.connect(self.update_amplitude_label)
        amplitude_signal_layout.addWidget(self.amplitude_slider)
        amplitude_signal_layout.addWidget(self.amplitude_value)

        # Add phase slider
        self.phase_signal_label = QLabel("Phase")
        phase_signal_layout = QHBoxLayout()
        self.phase_slider = QSlider(Qt.Horizontal)
        self.phase_slider.setRange(0, 360)
        self.phase_slider.setValue(0)
        self.phase_value = QLabel("0°")
        self.phase_slider.valueChanged.connect(self.update_phase_label)
        phase_signal_layout.addWidget(self.phase_signal_label)
        phase_signal_layout.addWidget(self.phase_slider)
        phase_signal_layout.addWidget(self.phase_value)

        self.add_signal_button = QPushButton("Add")
        self.add_signal_button.clicked.connect(self.add_component)
        self.save_signal_button = QPushButton("Save signal")
        self.save_signal_button.clicked.connect(self.save_signal)
        mixer_buttons_layout = QHBoxLayout()

        mixer_buttons_layout.addWidget(self.add_signal_button)
        mixer_buttons_layout.addWidget(self.save_signal_button)

        # Remove components
        self.remove_signal_label = QLabel("Remove signal components")
        remove_layout = QHBoxLayout()
        self.remove_signals_menu = QComboBox()
        self.remove_button = QPushButton("Remove")
        remove_layout.addWidget(self.remove_signals_menu)
        remove_layout.addWidget(self.remove_button)

        # Adjusting sampling frequency
        self.sampling_frequency_label = QLabel("Sampling frequency")
        sampling_freq_layout = QHBoxLayout()
        freq_value_layout = QVBoxLayout()
        self.sampling_freq_slider = QSlider(Qt.Horizontal)
        self.sampling_freq_slider.setRange(1, 150)
        self.sampling_freq_slider.setValue(1)
        self.sampling_freq_slider.valueChanged.connect(self.update_sampling_frequency_label)
        self.sampling_freq_slider.valueChanged.connect(self.queue_plot_update)
        self.freq_ratio_label = QLabel("0")
        self.sampling_freq_value = QLabel("1 Hz")
        freq_value_layout.addWidget(self.freq_ratio_label)
        freq_value_layout.addWidget(self.sampling_freq_value)
        sampling_freq_layout.addWidget(self.sampling_freq_slider)
        sampling_freq_layout.addLayout(freq_value_layout)
        self.remove_button.clicked.connect(self.remove_component)

        # Noise addition
        self.noise_label = QLabel("Noise addition")
        snr_layout = QHBoxLayout()
        self.snr_label = QLabel("SNR")
        self.snr_slider = QSlider(Qt.Horizontal)
        self.snr_slider.setRange(1, 50)
        self.snr_slider.setValue(50)
        self.snr_slider.valueChanged.connect(self.queue_plot_update)
        self.snr_slider.valueChanged.connect(self.update_snr_value)
        self.snr_value = QLabel("50")
        snr_layout.addWidget(self.snr_label)
        snr_layout.addWidget(self.snr_slider)
        snr_layout.addWidget(self.snr_value)

        controls_layout1.addWidget(self.upload_signal_label)
        controls_layout1.addWidget(self.upload_signal_button)
        controls_layout1.addLayout(reconstruction_method_layout)
        controls_layout1.addWidget(self.signal_mixer_label)
        controls_layout1.addWidget(self.freq_signal_label)
        controls_layout1.addLayout(freq_signal_layout)
        controls_layout1.addWidget(self.amplitude_signal_label)
        controls_layout1.addLayout(amplitude_signal_layout)
        controls_layout1.addLayout(phase_signal_layout)
        controls_layout1.addLayout(mixer_buttons_layout)

        controls_layout2.addWidget(self.remove_signal_label)
        controls_layout2.addLayout(remove_layout)
        controls_layout2.addWidget(self.sampling_frequency_label)
        controls_layout2.addLayout(sampling_freq_layout)
        controls_layout2.addWidget(self.noise_label)
        controls_layout2.addLayout(snr_layout)

        self.delete_signal_button = QPushButton("Delete")
        self.delete_signal_button.clicked.connect(self.delete_signal)
        controls_layout2.addWidget(self.delete_signal_button)

        container.addLayout(controls_layout1, 0, 0)
        container.addLayout(controls_layout2, 1, 0)

        # STYLING
        self.upload_signal_label.setObjectName("bold")
        self.signal_mixer_label.setObjectName("bold")
        self.remove_signal_label.setObjectName("bold")
        self.noise_label.setObjectName("bold")
        self.reconstruction_label.setObjectName("bold")
        self.upload_signal_button.setMaximumWidth(140)
        self.delete_signal_button.setMaximumWidth(140)
        self.add_signal_button.setMaximumWidth(120)
        self.save_signal_button.setMaximumWidth(120)
        self.remove_button.setMaximumWidth(120)
        self.freq_slider.setObjectName("custom_slider")  # Add style for sliders
        self.amplitude_slider.setObjectName("custom_slider")
        self.phase_slider.setObjectName("custom_slider")
        self.sampling_freq_slider.setObjectName("custom_slider")
        self.snr_slider.setObjectName("custom_slider")

        self.setStyleSheet("""
            QLabel{
                font-size:17px;
            }
            QLabel#bold{
                font-weight:500;
                font-size:20px;
             }
            QPushButton{
                font-size:17px;
                padding:5px 10px;
                border:3px solid grey;
                border-radius:15px;
                background-color:grey;
             }
            QComboBox{
                font-size:17px;
            }
            QSlider#custom_slider::groove:horizontal {
                border: 1px solid #bbb;
                background: #eee;
                height: 8px;
                border-radius: 4px;
            }
            QSlider#custom_slider::handle:horizontal {
                background: #4169E1;
                border: 1px solid #4169E1;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.update_sampling_frequency_label(self.sampling_freq_slider.value()) # call the update to init

    def create_graph_canvas(self, title="Graph"):
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setLabel('left', 'Amplitude')
        plot_widget.setLabel('bottom', 'Time' if title != "Frequency Domain" else "Frequency")
        plot_widget.addLegend()
        return plot_widget

    def create_error_canvas(self, title="Graph"):
        plot_widget = pg.PlotWidget(title=title)
        plot_widget.setYRange(-4, 4)
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setLabel('left', 'Amplitude')
        plot_widget.setLabel('bottom', 'Time' if title != "Frequency Domain" else "Frequency")
        plot_widget.addLegend()
        return plot_widget

    def upload_and_plot_signal(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Signal File", "", "CSV Files (*.csv)")
        if file_path:
            data = pd.read_csv(file_path)
            self.time = np.array(data.iloc[:, 0])
            self.amplitude = np.array(data.iloc[:, 1])
            self.original_signal = self.amplitude.copy()
            time_interval = self.time[1] - self.time[0]
            self.original_fs = 1 / time_interval if time_interval > 0 else 1000
            self.canvas1.clear()
            self.canvas1.plot(self.time, self.amplitude, pen='r', name="Original Signal")
            self.components = [] # clear the current components
            self.composer_frequencies = [] # clear the frequencies list
            self.calculate_max_freq()
            self.update_plots()
        self.remove_signals_menu.clear()

    def sample_signal(self, signal, sampling_rate):
        self.fs = sampling_rate
        if not hasattr(self, 'original_fs') or self.original_fs <= 0:
            print("Error: Original sampling frequency (self.original_fs) is not set or invalid.")
            return [], []
        sample_factor = self.original_fs / self.fs
        if sample_factor < 1:
            sample_factor = 1
        sample_indices = np.arange(0, len(signal), sample_factor).astype(int)
        sampled_signal = signal[sample_indices]
        sampled_times = self.time[sample_indices]
        # print(
        #     f"Slider Frequency: {self.fs} Hz, Downsampling Factor: {sample_factor}, Points Sampled: {len(sample_indices)}")
        return sampled_times, sampled_signal

    def reconstruct_whittaker_shannon(self, sampled_times, sampled_signal):
        t = np.linspace(self.time[0], self.time[-1], len(self.time))
        reconstruction = np.zeros_like(t)
        sampled_times = np.array(sampled_times)
        sampled_signal = np.array(sampled_signal)
        sample_interval = 1 / self.fs

        for i, ti in enumerate(sampled_times):
            reconstruction += sampled_signal[i] * np.sinc((t - ti) / sample_interval)

        if len(sampled_times) > 0:
            last_sample_value = sampled_signal[-1]
            reconstruction[t > sampled_times[-1]] = last_sample_value
        return t, reconstruction

    def reconstruct_zoh(self, sampled_times, sampled_signal):
        t = np.linspace(self.time[0], self.time[-1], len(self.time))
        reconstruction = np.zeros_like(t)
        sampled_times = np.array(sampled_times)
        sampled_signal = np.array(sampled_signal)
        for i in range(len(sampled_times) - 1):
            mask = (t >= sampled_times[i]) & (t < sampled_times[i + 1])
            reconstruction[mask] = sampled_signal[i]
        reconstruction[t >= sampled_times[-1]] = sampled_signal[-1]
        return t, reconstruction

    def reconstruct_linear(self, sampled_times, sampled_signal):
        t = np.linspace(self.time[0], self.time[-1], len(self.time))
        return t, np.interp(t, sampled_times, sampled_signal)

    def queue_plot_update(self):
        """Queues an update of the plots using a timer."""
        self.update_timer.start(200)  # Delay of 200ms

    def update_plots(self):
        self.update_timer.stop()
        if not hasattr(self, 'amplitude'):
            return

        # Get SNR value
        snr = self.snr_slider.value()

        # Add noise to a copy of the original signal
        noisy_signal = self.add_noise_to_signal(self.amplitude.copy(), snr)

        # Sampling
        sampling_rate = self.sampling_freq_slider.value()
        sampled_times, sampled_signal = self.sample_signal(noisy_signal, sampling_rate)

        # Reconstruction method
        reconstruction_method = self.reconstruction_menu.currentText()
        if reconstruction_method == "Whittaker-Shannon":
            t_reconstructed, reconstructed_signal = self.reconstruct_whittaker_shannon(sampled_times, sampled_signal)
        elif reconstruction_method == "Zero-Order Hold":
            t_reconstructed, reconstructed_signal = self.reconstruct_zoh(sampled_times, sampled_signal)
        elif reconstruction_method == "Linear Interpolation":
            t_reconstructed, reconstructed_signal = self.reconstruct_linear(sampled_times, sampled_signal)

        # Plotting on canvas1
        self.canvas1.clear()
        self.canvas1.plot(self.time, noisy_signal, pen='r',
                          name="Noisy Signal" if self.snr_slider.value() > 0 else "Original Signal")
        self.canvas1.plot(sampled_times, sampled_signal, pen=None, symbol='o', symbolBrush="b", name="Sampled Points")

        # Plotting on canvas2
        self.canvas2.clear()
        self.canvas2.plot(t_reconstructed, reconstructed_signal, pen='g', name="Reconstructed Signal")

        # Error calculation and plotting
        if hasattr(self, 'original_signal'):
            error_signal = self.original_signal - np.interp(self.time, t_reconstructed,
                                                            reconstructed_signal)  # Use the original (without noise)
            self.canvas3.clear()
            self.canvas3.plot(self.time, error_signal, pen='r', name="Error Signal")
        else:
            self.canvas3.clear()
        # Frequency domain plot
        self.plot_frequency_domain()

    def plot_frequency_domain(self):
        if not hasattr(self, 'amplitude'):
            return

        time_step = np.mean(np.diff(self.time))
        n = len(self.amplitude)
        fft_values = fft(self.amplitude)
        frequencies = fftfreq(n, d=time_step)
        magnitude = np.abs(fft_values) / n

        self.canvas4.clear()
        self.canvas4.plot(frequencies, magnitude, pen='m', name="Original Spectrum")

        if hasattr(self, 'fs'):
            num_repeats = 2
            bandwidth = self.fs
            colors = ['#8B0000', '#006400', '#00008B'] #Dim versions of red, green and blue
            for i in range(1, num_repeats + 1):
                band_shift = i * bandwidth
                color = colors[i - 1] if i-1 < len(colors) else '#4F4F4F'  # Default to a dim gray if no color available
                self.canvas4.plot(frequencies + band_shift, magnitude, pen=color, name=f"Replica {i}")
                self.canvas4.plot(frequencies - band_shift, magnitude, pen=color, name=f"Replica -{i}")


            max_freq_to_display = bandwidth * (
                    num_repeats + 1)  # Set dynamic Range to show the replicas and the original
            self.canvas4.setLabel('bottom', 'Frequency (Hz)')
            self.canvas4.setXRange(-max_freq_to_display, max_freq_to_display)  # Set fixed range
            self.canvas4.setLabel('left', 'Magnitude')

    def update_frequency_label(self, value):
        self.frequency_value.setText(f"{value} Hz")

    def update_amplitude_label(self, value):
        self.amplitude_value.setText(f"{value}")

    def update_phase_label(self, value):
        self.phase_value.setText(f"{value}°")

    def add_component(self):
        frequency = self.freq_slider.value()
        amplitude = self.amplitude_slider.value()
        phase = self.phase_slider.value()
        self.composer_frequencies.append(frequency)
        self.components.append((frequency, amplitude, phase))
        component_text = f"Frequency: {frequency} Hz, Amplitude: {amplitude}, Phase: {phase}°"
        self.remove_signals_menu.addItem(component_text)
        self.update_composite_signal()

    def update_composite_signal(self, duration=2, sample_rate=1000):
        self.time = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        self.amplitude = np.zeros(len(self.time))
        self.original_signal = np.zeros(len(self.time))
        time_interval = self.time[1] - self.time[0]
        self.original_fs = 1 / time_interval if time_interval > 0 else 1000

        for frequency, amplitude, phase in self.components:
            phase_rad = np.deg2rad(phase)
            component_signal = amplitude * np.cos(2 * np.pi * frequency * self.time + phase_rad)
            self.amplitude += component_signal
            self.original_signal += component_signal

        self.calculate_max_freq() #Calculate max_freq after signal changes
        self.canvas1.clear()
        self.canvas2.clear()
        self.canvas3.clear()
        self.canvas4.clear()
        self.freq_slider.setValue(0)
        self.amplitude_slider.setValue(0)
        self.phase_slider.setValue(0)
        self.canvas1.plot(self.time, self.amplitude, pen='r', name="Original Signal")
        self.update_plots()

    def calculate_max_freq(self):
        """
        Calculate the maximum frequency consistently for both loaded and composed signals.
        Prioritizes loaded signal, falls back to composite signal components.
        """
        if hasattr(self, 'amplitude') and len(self.amplitude) > 0:
            # Method 1: For loaded signals - use FFT to find dominant frequency
            time_interval = self.time[1] - self.time[0]
            fft_values = np.abs(fft(self.amplitude))
            freqs = fftfreq(len(self.amplitude), time_interval)
            loaded_max_freq = freqs[np.argmax(fft_values[:len(freqs) // 2])]
            self.max_freq = loaded_max_freq

        # Always consider composer frequencies for max frequency
        if hasattr(self, 'composer_frequencies') and len(self.composer_frequencies) > 0:
            composite_max_freq = max(self.composer_frequencies)

            # If both loaded and composed signals exist, take the maximum
            if hasattr(self, 'max_freq'):
                self.max_freq = max(self.max_freq, composite_max_freq)
            else:
                self.max_freq = composite_max_freq

        # Default to 0 if no frequencies found
        if not hasattr(self, 'max_freq'):
            self.max_freq = 0

        # Always update the sampling frequency label after calculation
        self.update_sampling_frequency_label(self.sampling_freq_slider.value())

    def update_sampling_frequency_label(self, value):
        """
        Update the sampling frequency label with a normalized representation.
        Works consistently for both loaded and composed signals.
        """
        # Store the actual sampling frequency
        sampling_freq = value

        # Ensure max_freq is calculated and available
        if not hasattr(self, 'max_freq'):
            self.calculate_max_freq()

        # Only set ratio and update label if max_freq is greater than 0
        if self.max_freq > 0:
            # Normalized representation
            freq_ratio = sampling_freq / self.max_freq
            self.freq_ratio_label.setText(f"{freq_ratio:.2f} Fmax")
            self.sampling_freq_value.setText(f"{sampling_freq} Hz")
        else:
            # Fallback if no max frequency is known
            self.freq_ratio_label.setText("0")
            self.sampling_freq_value.setText(f"{sampling_freq} Hz")

    def update_snr_value(self, value):
        self.snr_value.setText(f"{value}")

    def add_noise_to_signal(self, signal, snr):
        if snr > 0:
            signal_power = np.mean(signal ** 2)
            snr_linear = 10 ** (snr / 10)
            noise_power = signal_power / snr_linear
            noise = np.sqrt(noise_power) * np.random.normal(size=len(signal))
            noisy_signal = signal + noise
            return noisy_signal
        return signal

    def remove_component(self):
        index = self.remove_signals_menu.currentIndex()
        if index >= 0:
            removed_component = list(self.components[index])
            del self.components[index]
            self.remove_signals_menu.removeItem(index)
            self.composer_frequencies.remove(removed_component[0])
            if len(self.composer_frequencies) > 0:
                self.calculate_max_freq() # Calculate max_freq after removing components
            self.update_composite_signal()
            self.update_sampling_frequency_label(self.sampling_freq_slider.value())
        if len(self.composer_frequencies) == 0:
            self.delete_signal()

    def delete_signal(self):
        self.initUI()
        self.amplitude = np.array([])
        self.time = np.array([])
        self.frequencies = np.array([])
        self.reconstructed_signal = np.array([])
        self.max_freq = 0
        self.components = []
        self.generated_signal = None
        self.composer_frequencies = []

    def save_signal(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if filename:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Time', 'Amplitude'])
                for a, b in zip(self.time, self.amplitude):
                    writer.writerow([a, b])


app = QApplication(sys.argv)
window = SamplingTheorem()
window.show()
sys.exit(app.exec_())