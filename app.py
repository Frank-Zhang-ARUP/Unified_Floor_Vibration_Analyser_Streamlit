import io
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
from scipy.io import wavfile


st.set_page_config(
    page_title="Floor Vibration Test Analyser",
    layout="wide",
)

st.title("Floor Vibration Test Analyser")
st.caption(
    "Unified analysis for background vibration, heel-drop, walking, running, "
    "jumping and other floor-activity WAV measurements."
)


@dataclass
class SignalRecord:
    name: str
    sample_rate: float
    acceleration: np.ndarray


@dataclass
class BackgroundRecord:
    sample_rate: float
    acceleration: np.ndarray
    rms: float
    peak: float
    psd_frequency: np.ndarray
    psd_db: np.ndarray
    spectrum_frequency: np.ndarray
    spectrum_amplitude: np.ndarray


with st.sidebar:
    st.header("Analysis mode")
    analysis_mode = st.selectbox(
        "Test type",
        ["Heel Drop", "Walking", "Running", "Jumping", "Other Activity"],
    )

    st.header("Calibration")
    sensitivity_mv = st.number_input(
        "Sensor sensitivity [mV/(m/s²)]",
        min_value=0.001,
        max_value=10000.0,
        value=10.2,
        step=0.1,
    )
    chain_gain = st.number_input(
        "Recording-chain gain",
        min_value=0.000001,
        max_value=1000000.0,
        value=1.0,
        step=0.1,
    )
    full_scale_v = st.number_input(
        "WAV full-scale [Vpeak]",
        min_value=0.000001,
        max_value=1000.0,
        value=1.0,
        step=0.1,
    )
    samples_are_acceleration = st.checkbox(
        "WAV samples already represent m/s²",
        value=False,
    )

    st.header("Signal processing")
    high_pass_hz = st.number_input(
        "High-pass frequency [Hz], 0 = off",
        min_value=0.0,
        max_value=100.0,
        value=0.5,
        step=0.1,
    )
    low_pass_hz = st.number_input(
        "Low-pass frequency [Hz], 0 = off",
        min_value=0.0,
        max_value=5000.0,
        value=100.0,
        step=1.0,
    )
    frequency_min_hz = st.number_input(
        "Analysis frequency minimum [Hz]",
        min_value=0.1,
        max_value=100.0,
        value=0.5,
        step=0.1,
    )
    frequency_max_hz = st.number_input(
        "Analysis frequency maximum [Hz]",
        min_value=1.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
    )
    prominence_db = st.number_input(
        "Minimum spectral prominence [dB]",
        min_value=1.0,
        max_value=40.0,
        value=5.0,
        step=0.5,
    )

    st.header("Heel-drop settings")
    detection_sigma = st.number_input(
        "Impact threshold [robust sigma]",
        min_value=2.0,
        max_value=30.0,
        value=7.0,
        step=0.5,
    )
    minimum_event_gap_s = st.number_input(
        "Minimum gap between impacts [s]",
        min_value=0.3,
        max_value=20.0,
        value=2.0,
        step=0.1,
    )
    impact_transient_s = st.number_input(
        "Exclude impact transient [s]",
        min_value=0.0,
        max_value=1.0,
        value=0.08,
        step=0.01,
    )
    decay_window_s = st.number_input(
        "Free-decay window [s]",
        min_value=0.5,
        max_value=20.0,
        value=4.0,
        step=0.25,
    )


background_file = st.file_uploader(
    "Background WAV, optional but recommended",
    type=["wav"],
    key="background_wav",
)

test_files = st.file_uploader(
    "Test WAV file(s)",
    type=["wav"],
    accept_multiple_files=True,
    key="test_wavs",
)


def read_wav(uploaded_file) -> tuple[float, np.ndarray]:
    sample_rate, samples = wavfile.read(io.BytesIO(uploaded_file.getvalue()))

    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    if np.issubdtype(samples.dtype, np.integer):
        limits = np.iinfo(samples.dtype)
        denominator = max(abs(limits.min), limits.max)
        samples = samples.astype(np.float64) / denominator
    else:
        samples = samples.astype(np.float64)

    return float(sample_rate), samples


def convert_to_acceleration(samples: np.ndarray) -> np.ndarray:
    if samples_are_acceleration:
        return samples

    volts = samples * full_scale_v
    sensitivity_v = sensitivity_mv / 1000.0
    return volts / (sensitivity_v * chain_gain)


def condition_signal(acceleration: np.ndarray, sample_rate: float) -> np.ndarray:
    conditioned = signal.detrend(acceleration, type="constant")

    if 0 < high_pass_hz < sample_rate / 2:
        high_pass = signal.butter(
            4,
            high_pass_hz,
            btype="highpass",
            fs=sample_rate,
            output="sos",
        )
        conditioned = signal.sosfiltfilt(high_pass, conditioned)

    if 0 < low_pass_hz < sample_rate / 2:
        low_pass = signal.butter(
            4,
            low_pass_hz,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        conditioned = signal.sosfiltfilt(low_pass, conditioned)

    return conditioned


def calculate_peak_spectrum(
    acceleration: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = len(acceleration)
    window = signal.windows.hann(sample_count, sym=False)
    coherent_gain = np.mean(window)
    transformed = np.fft.rfft((acceleration - np.mean(acceleration)) * window)
    amplitude = np.abs(transformed) / (sample_count * coherent_gain)

    if len(amplitude) > 2:
        amplitude[1:-1] *= 2.0

    frequency = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate)
    return frequency, amplitude


def calculate_psd(
    acceleration: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    segment_length = min(
        len(acceleration),
        max(256, int(sample_rate * 2.0)),
    )
    frequency, density = signal.welch(
        acceleration,
        fs=sample_rate,
        window="hann",
        nperseg=segment_length,
        noverlap=segment_length // 2,
        detrend="constant",
        scaling="density",
    )
    density_db = 10.0 * np.log10(
        np.maximum(density, np.finfo(float).tiny)
    )
    return frequency, density_db


def detect_heel_drops(
    acceleration: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    envelope = np.abs(signal.hilbert(acceleration))
    smoothing_samples = max(1, int(0.02 * sample_rate))
    kernel = np.ones(smoothing_samples) / smoothing_samples
    smoothed = np.convolve(envelope, kernel, mode="same")

    median = np.median(smoothed)
    robust_sigma = (
        1.4826 * np.median(np.abs(smoothed - median))
        + np.finfo(float).eps
    )
    threshold = median + detection_sigma * robust_sigma

    impact_indices, _ = signal.find_peaks(
        smoothed,
        height=threshold,
        distance=max(1, int(minimum_event_gap_s * sample_rate)),
    )

    refined_indices = []
    search_radius = max(1, int(0.15 * sample_rate))
    for index in impact_indices:
        lower = max(0, index - search_radius)
        upper = min(len(acceleration), index + search_radius)
        refined = lower + int(np.argmax(np.abs(acceleration[lower:upper])))
        refined_indices.append(refined)

    return np.asarray(refined_indices, dtype=int)


def identify_spectral_peaks(
    frequency: np.ndarray,
    psd_db: np.ndarray,
    background: BackgroundRecord | None,
) -> list[dict]:
    mask = (
        (frequency >= frequency_min_hz)
        & (frequency <= min(frequency_max_hz, frequency[-1]))
    )
    selected_frequency = frequency[mask]
    selected_psd = psd_db[mask]

    if len(selected_frequency) < 3:
        return []

    peak_indices, properties = signal.find_peaks(
        selected_psd,
        prominence=prominence_db,
    )

    if background is None:
        background_levels = np.full_like(selected_frequency, np.nan)
    else:
        background_levels = np.interp(
            selected_frequency,
            background.psd_frequency,
            background.psd_db,
            left=np.nan,
            right=np.nan,
        )

    peaks = []
    for position, peak_index in enumerate(peak_indices):
        background_level = background_levels[peak_index]
        level_above_background = (
            selected_psd[peak_index] - background_level
            if np.isfinite(background_level)
            else np.nan
        )
        peaks.append(
            {
                "Frequency_Hz": float(selected_frequency[peak_index]),
                "PSD_dB": float(selected_psd[peak_index]),
                "Prominence_dB": float(properties["prominences"][position]),
                "Above_background_dB": float(level_above_background),
            }
        )

    peaks.sort(key=lambda item: item["PSD_dB"], reverse=True)
    return peaks


def build_background(uploaded_file) -> BackgroundRecord | None:
    if uploaded_file is None:
        return None

    sample_rate, samples = read_wav(uploaded_file)
    acceleration = condition_signal(
        convert_to_acceleration(samples[:, 0]),
        sample_rate,
    )
    psd_frequency, psd_db = calculate_psd(acceleration, sample_rate)
    spectrum_frequency, spectrum_amplitude = calculate_peak_spectrum(
        acceleration,
        sample_rate,
    )

    return BackgroundRecord(
        sample_rate=sample_rate,
        acceleration=acceleration,
        rms=float(np.sqrt(np.mean(acceleration ** 2))),
        peak=float(np.max(np.abs(acceleration))),
        psd_frequency=psd_frequency,
        psd_db=psd_db,
        spectrum_frequency=spectrum_frequency,
        spectrum_amplitude=spectrum_amplitude,
    )


def make_plot_image(
    title: str,
    time: np.ndarray,
    acceleration: np.ndarray,
    spectrum_frequency: np.ndarray,
    spectrum_amplitude: np.ndarray,
    background: BackgroundRecord | None,
    impact_times: list[float],
) -> io.BytesIO:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].plot(time, acceleration, linewidth=0.7, label="Test")
    for impact_number, impact_time in enumerate(impact_times, start=1):
        axes[0].axvline(impact_time, color="red", alpha=0.4, linewidth=0.8)
        axes[0].text(
            impact_time,
            axes[0].get_ylim()[1],
            f"HD{impact_number}",
            fontsize=7,
            rotation=90,
            verticalalignment="top",
        )
    axes[0].set_title(f"{title} - Time History")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Acceleration [m/s²]")
    axes[0].grid(alpha=0.25)

    spectrum_mask = (
        (spectrum_frequency >= frequency_min_hz)
        & (spectrum_frequency <= min(frequency_max_hz, spectrum_frequency[-1]))
    )
    axes[1].plot(
        spectrum_frequency[spectrum_mask],
        spectrum_amplitude[spectrum_mask],
        linewidth=1.0,
        label="Test",
    )

    if background is not None:
        background_mask = (
            (background.spectrum_frequency >= frequency_min_hz)
            & (
                background.spectrum_frequency
                <= min(frequency_max_hz, background.spectrum_frequency[-1])
            )
        )
        axes[1].plot(
            background.spectrum_frequency[background_mask],
            background.spectrum_amplitude[background_mask],
            linewidth=1.0,
            alpha=0.8,
            label="Background",
        )

    axes[1].set_title("Peak Amplitude Spectrum")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Peak acceleration amplitude [m/s²]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    image = io.BytesIO()
    figure.savefig(image, format="png", dpi=160)
    plt.close(figure)
    image.seek(0)
    return image


def create_excel_report(
    summary: pd.DataFrame,
    heel_drop_events: pd.DataFrame,
    spectral_peaks: pd.DataFrame,
    plot_images: list[tuple[str, io.BytesIO]],
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        workbook = writer.book
        header_format = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#17365D",
                "font_color": "#FFFFFF",
                "border": 1,
            }
        )

        summary_sheet = writer.sheets["Summary"]
        summary_sheet.set_row(0, 24, header_format)
        summary_sheet.freeze_panes(1, 0)
        summary_sheet.set_column(0, len(summary.columns) - 1, 20)

        if not heel_drop_events.empty:
            heel_drop_events.to_excel(
                writer,
                index=False,
                sheet_name="Heel Drop Events",
            )
            event_sheet = writer.sheets["Heel Drop Events"]
            event_sheet.set_row(0, 24, header_format)
            event_sheet.freeze_panes(1, 0)
            event_sheet.set_column(0, len(heel_drop_events.columns) - 1, 20)

        if not spectral_peaks.empty:
            spectral_peaks.to_excel(
                writer,
                index=False,
                sheet_name="Spectral Peaks",
            )
            peak_sheet = writer.sheets["Spectral Peaks"]
            peak_sheet.set_row(0, 24, header_format)
            peak_sheet.freeze_panes(1, 0)
            peak_sheet.set_column(0, len(spectral_peaks.columns) - 1, 20)

        for chart_number, (chart_name, image) in enumerate(plot_images[:25], start=1):
            sheet = workbook.add_worksheet(f"Chart_{chart_number:02d}")
            sheet.write("A1", chart_name, header_format)
            image.seek(0)
            sheet.insert_image(
                "A3",
                f"chart_{chart_number:02d}.png",
                {
                    "image_data": image,
                    "x_scale": 0.72,
                    "y_scale": 0.72,
                },
            )

    return output.getvalue()


def add_dataframe_page(
    pdf: PdfPages,
    title: str,
    dataframe: pd.DataFrame,
) -> None:
    if dataframe.empty:
        return

    display_data = dataframe.copy().round(5)
    figure = plt.figure(figsize=(11.69, 8.27))
    axis = figure.add_axes([0.04, 0.06, 0.92, 0.86])
    axis.axis("off")
    table = axis.table(
        cellText=display_data.values,
        colLabels=display_data.columns,
        loc="upper left",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.25)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    pdf.savefig(figure)
    plt.close(figure)


def create_pdf_report(
    summary: pd.DataFrame,
    heel_drop_events: pd.DataFrame,
    spectral_peaks: pd.DataFrame,
    plot_images: list[tuple[str, io.BytesIO]],
    background: BackgroundRecord | None,
) -> bytes:
    output = io.BytesIO()

    with PdfPages(output) as pdf:
        cover = plt.figure(figsize=(11.69, 8.27))
        cover.text(
            0.06,
            0.90,
            "Floor Vibration Test Report",
            fontsize=20,
            fontweight="bold",
        )
        cover.text(0.06, 0.84, f"Analysis mode: {analysis_mode}", fontsize=12)
        cover.text(
            0.06,
            0.79,
            "Background treatment: "
            + (
                "PSD comparison and RMS energy correction"
                if background is not None
                else "No background WAV supplied"
            ),
            fontsize=11,
        )
        cover.text(
            0.06,
            0.70,
            "RMS correction: sqrt(test RMS² - background RMS²), clipped at zero.",
            fontsize=10,
        )
        cover.text(
            0.06,
            0.66,
            "Peak acceleration is reported as measured and is not background-subtracted.",
            fontsize=10,
        )
        cover.text(
            0.06,
            0.62,
            "Spectral candidates include their level above the background PSD.",
            fontsize=10,
        )
        pdf.savefig(cover)
        plt.close(cover)

        add_dataframe_page(pdf, "Summary", summary)
        add_dataframe_page(pdf, "Heel Drop Events", heel_drop_events)
        add_dataframe_page(pdf, "Spectral Peaks", spectral_peaks)

        for chart_name, image in plot_images:
            image.seek(0)
            image_array = plt.imread(image)
            figure = plt.figure(figsize=(11.69, 8.27))
            axis = figure.add_subplot(111)
            axis.imshow(image_array)
            axis.axis("off")
            figure.suptitle(chart_name, fontsize=14, fontweight="bold")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    return output.getvalue()


background = build_background(background_file)

if background is not None:
    st.success(
        "Background loaded: "
        f"RMS = {background.rms * 1000:.3f} mm/s²; "
        f"peak = {background.peak * 1000:.3f} mm/s²."
    )

if not test_files:
    st.info(
        "Upload one or more test WAV files. For absolute acceleration, confirm "
        "sensor sensitivity, recording-chain gain and WAV full-scale voltage."
    )
else:
    summary_rows = []
    heel_drop_rows = []
    spectral_peak_rows = []
    report_images = []

    for uploaded_test in test_files:
        sample_rate, samples = read_wav(uploaded_test)

        for channel_index in range(samples.shape[1]):
            record_name = f"{uploaded_test.name} | Ch {channel_index + 1}"
            acceleration = condition_signal(
                convert_to_acceleration(samples[:, channel_index]),
                sample_rate,
            )
            time = np.arange(len(acceleration)) / sample_rate

            measured_peak = float(np.max(np.abs(acceleration)))
            measured_rms = float(np.sqrt(np.mean(acceleration ** 2)))

            if background is None:
                corrected_rms = np.nan
            else:
                corrected_rms = math.sqrt(
                    max(0.0, measured_rms ** 2 - background.rms ** 2)
                )

            spectrum_frequency, spectrum_amplitude = calculate_peak_spectrum(
                acceleration,
                sample_rate,
            )
            psd_frequency, psd_db = calculate_psd(acceleration, sample_rate)
            peaks = identify_spectral_peaks(
                psd_frequency,
                psd_db,
                background,
            )

            impact_indices = np.asarray([], dtype=int)
            impact_times = []
            event_frequencies = []

            if analysis_mode == "Heel Drop":
                impact_indices = detect_heel_drops(acceleration, sample_rate)
                impact_times = [float(index / sample_rate) for index in impact_indices]

                for event_number, impact_index in enumerate(
                    impact_indices,
                    start=1,
                ):
                    decay_start = impact_index + int(impact_transient_s * sample_rate)
                    available_before_next = max(0.5, minimum_event_gap_s - 0.1)
                    decay_end = min(
                        len(acceleration),
                        decay_start + int(decay_window_s * sample_rate),
                        impact_index + int(available_before_next * sample_rate),
                    )

                    if decay_end - decay_start < 128:
                        continue

                    decay = acceleration[decay_start:decay_end]
                    decay_frequency, decay_psd_db = calculate_psd(decay, sample_rate)
                    decay_peaks = identify_spectral_peaks(
                        decay_frequency,
                        decay_psd_db,
                        background,
                    )

                    dominant_event_frequency = (
                        decay_peaks[0]["Frequency_Hz"]
                        if decay_peaks
                        else np.nan
                    )
                    if np.isfinite(dominant_event_frequency):
                        event_frequencies.append(dominant_event_frequency)

                    peak_lower = max(0, impact_index - int(0.15 * sample_rate))
                    peak_upper = min(
                        len(acceleration),
                        impact_index + int(0.5 * sample_rate),
                    )
                    event_peak = float(
                        np.max(np.abs(acceleration[peak_lower:peak_upper]))
                    )

                    heel_drop_rows.append(
                        {
                            "File_channel": record_name,
                            "Event": event_number,
                            "Event_time_s": impact_index / sample_rate,
                            "Peak_acc_m_s2": event_peak,
                            "Peak_acc_mm_s2": event_peak * 1000.0,
                            "Dominant_frequency_Hz": dominant_event_frequency,
                            "Above_background_dB": (
                                decay_peaks[0]["Above_background_dB"]
                                if decay_peaks
                                else np.nan
                            ),
                        }
                    )

            if event_frequencies:
                dominant_response_frequency = float(np.median(event_frequencies))
            elif peaks:
                dominant_response_frequency = peaks[0]["Frequency_Hz"]
            else:
                search_mask = (
                    (psd_frequency >= frequency_min_hz)
                    & (psd_frequency <= min(frequency_max_hz, psd_frequency[-1]))
                )
                if np.any(search_mask):
                    selected_indices = np.where(search_mask)[0]
                    dominant_index = selected_indices[np.argmax(psd_db[search_mask])]
                    dominant_response_frequency = float(
                        psd_frequency[dominant_index]
                    )
                else:
                    dominant_response_frequency = np.nan

            dominant_above_background = (
                peaks[0]["Above_background_dB"] if peaks else np.nan
            )

            summary_rows.append(
                {
                    "Mode": analysis_mode,
                    "File_channel": record_name,
                    "Sample_rate_Hz": sample_rate,
                    "Duration_s": len(acceleration) / sample_rate,
                    "Measured_peak_m_s2": measured_peak,
                    "Measured_peak_mm_s2": measured_peak * 1000.0,
                    "Measured_peak_pct_g": measured_peak / 9.80665 * 100.0,
                    "Measured_RMS_m_s2": measured_rms,
                    "Background_RMS_m_s2": (
                        background.rms if background is not None else np.nan
                    ),
                    "Background_corrected_RMS_m_s2": corrected_rms,
                    "Dominant_response_Hz": dominant_response_frequency,
                    "Dominant_above_background_dB": dominant_above_background,
                    "Detected_heel_drops": (
                        len(event_frequencies)
                        if analysis_mode == "Heel Drop"
                        else np.nan
                    ),
                }
            )

            for peak_rank, peak in enumerate(peaks[:10], start=1):
                spectral_peak_rows.append(
                    {
                        "File_channel": record_name,
                        "Rank": peak_rank,
                        **peak,
                    }
                )

            plot_image = make_plot_image(
                record_name,
                time,
                acceleration,
                spectrum_frequency,
                spectrum_amplitude,
                background,
                impact_times,
            )
            report_images.append((record_name, plot_image))

            st.header(record_name)
            metric_1, metric_2, metric_3, metric_4 = st.columns(4)
            metric_1.metric(
                "Measured peak acceleration",
                f"{measured_peak * 1000.0:.3f} mm/s²",
            )
            metric_2.metric(
                "Measured RMS",
                f"{measured_rms * 1000.0:.3f} mm/s²",
            )
            metric_3.metric(
                "Background-corrected RMS",
                (
                    f"{corrected_rms * 1000.0:.3f} mm/s²"
                    if np.isfinite(corrected_rms)
                    else "N/A"
                ),
            )
            metric_4.metric(
                "Dominant response frequency",
                (
                    f"{dominant_response_frequency:.2f} Hz"
                    if np.isfinite(dominant_response_frequency)
                    else "N/A"
                ),
            )

            time_figure = go.Figure()
            time_figure.add_trace(
                go.Scatter(
                    x=time,
                    y=acceleration,
                    name="Test",
                    line={"width": 1},
                )
            )

            for event_number, impact_time in enumerate(impact_times, start=1):
                time_figure.add_vline(
                    x=impact_time,
                    line_width=1,
                    line_dash="dot",
                    line_color="red",
                )
                time_figure.add_annotation(
                    x=impact_time,
                    y=1.0,
                    yref="paper",
                    text=f"HD{event_number}",
                    showarrow=False,
                    textangle=-90,
                    yanchor="top",
                )

            time_figure.update_layout(
                title=f"{analysis_mode} - Time History",
                xaxis_title="Time [s]",
                yaxis_title="Acceleration [m/s²]",
                height=380,
            )
            st.plotly_chart(time_figure, use_container_width=True)

            spectrum_mask = (
                (spectrum_frequency >= frequency_min_hz)
                & (
                    spectrum_frequency
                    <= min(frequency_max_hz, spectrum_frequency[-1])
                )
            )

            spectrum_figure = go.Figure()
            spectrum_figure.add_trace(
                go.Scatter(
                    x=spectrum_frequency[spectrum_mask],
                    y=spectrum_amplitude[spectrum_mask],
                    name="Test",
                    line={"width": 1},
                )
            )

            if background is not None:
                background_mask = (
                    (background.spectrum_frequency >= frequency_min_hz)
                    & (
                        background.spectrum_frequency
                        <= min(
                            frequency_max_hz,
                            background.spectrum_frequency[-1],
                        )
                    )
                )
                spectrum_figure.add_trace(
                    go.Scatter(
                        x=background.spectrum_frequency[background_mask],
                        y=background.spectrum_amplitude[background_mask],
                        name="Background",
                        line={"width": 1},
                    )
                )

            spectrum_figure.update_layout(
                title="Peak Amplitude Spectrum",
                xaxis_title="Frequency [Hz]",
                yaxis_title="Peak acceleration amplitude [m/s²]",
                height=380,
            )
            st.plotly_chart(spectrum_figure, use_container_width=True)

            peak_table = pd.DataFrame(peaks[:10])
            st.subheader("Major spectral peaks")
            if peak_table.empty:
                st.info("No peak met the selected prominence threshold.")
            else:
                st.dataframe(peak_table, use_container_width=True)

    summary_dataframe = pd.DataFrame(summary_rows)
    heel_drop_dataframe = pd.DataFrame(heel_drop_rows)
    spectral_peak_dataframe = pd.DataFrame(spectral_peak_rows)

    st.header("Combined results")
    st.dataframe(summary_dataframe, use_container_width=True)

    if not heel_drop_dataframe.empty:
        st.subheader("Heel-drop event results")
        st.dataframe(heel_drop_dataframe, use_container_width=True)

    excel_bytes = create_excel_report(
        summary_dataframe,
        heel_drop_dataframe,
        spectral_peak_dataframe,
        report_images,
    )
    pdf_bytes = create_pdf_report(
        summary_dataframe,
        heel_drop_dataframe,
        spectral_peak_dataframe,
        report_images,
        background,
    )

    download_1, download_2 = st.columns(2)
    with download_1:
        st.download_button(
            "Download Excel report with charts",
            data=excel_bytes,
            file_name="floor_vibration_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with download_2:
        st.download_button(
            "Download PDF report with charts",
            data=pdf_bytes,
            file_name="floor_vibration_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.warning(
        "Background correction is applied only to RMS energy using "
        "sqrt(test RMS² - background RMS²). Peak acceleration remains the "
        "measured value. Spectral peaks are assessed by dB above background PSD."
    )
