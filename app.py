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

st.set_page_config(page_title="Arup Acoustics | Floor Vibration Analyser", layout="wide")
st.title("Arup Acoustics")
st.subheader("Floor Vibration Test Analyser")
st.caption("Unified analysis of background, heel drop, walking, running and jumping WAV measurements.")

TEST_ORDER = ["Background", "Heel Drop", "Walking", "Running", "Jumping"]
ACTIVITY_KEYS = ["Walking", "Running", "Jumping"]

@dataclass
class BackgroundRecord:
    rms: float
    peak: float
    psd_frequency: np.ndarray
    psd_db: np.ndarray
    spectrum_frequency: np.ndarray
    spectrum_amplitude: np.ndarray

with st.sidebar:
    st.header("Calibration")
    sensitivity_mv = st.number_input("Sensor sensitivity [mV/(m/s²)]", 0.001, 10000.0, 10.2, 0.1)
    chain_gain = st.number_input("Recording-chain gain", 0.000001, 1000000.0, 1.0, 0.1)
    full_scale_v = st.number_input("WAV full-scale [Vpeak]", 0.000001, 1000.0, 1.0, 0.1)
    samples_are_acceleration = st.checkbox("WAV samples already represent m/s²", False)
    st.header("Signal processing")
    high_pass_hz = st.number_input("High-pass [Hz], 0 = off", 0.0, 100.0, 0.5, 0.1)
    low_pass_hz = st.number_input("Low-pass [Hz], 0 = off", 0.0, 5000.0, 100.0, 1.0)
    frequency_min_hz = st.number_input("Frequency minimum [Hz]", 0.1, 100.0, 0.5, 0.1)
    frequency_max_hz = st.number_input("Frequency maximum [Hz]", 1.0, 1000.0, 80.0, 1.0)
    prominence_db = st.number_input("Minimum spectral prominence [dB]", 1.0, 40.0, 5.0, 0.5)
    st.header("Heel-drop settings")
    detection_sigma = st.number_input("Impact threshold [robust sigma]", 2.0, 30.0, 7.0, 0.5)
    minimum_event_gap_s = st.number_input("Minimum gap [s]", 0.3, 20.0, 2.0, 0.1)
    impact_transient_s = st.number_input("Exclude impact transient [s]", 0.0, 1.0, 0.08, 0.01)
    decay_window_s = st.number_input("Free-decay window [s]", 0.5, 20.0, 4.0, 0.25)

st.markdown("### Upload one WAV for each available test")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    background_file = st.file_uploader("Background", type=["wav"], key="background")
with c2:
    heel_drop_file = st.file_uploader("Heel Drop", type=["wav"], key="heel")
with c3:
    walking_file = st.file_uploader("Walking", type=["wav"], key="walk")
with c4:
    running_file = st.file_uploader("Running", type=["wav"], key="run")
with c5:
    jumping_file = st.file_uploader("Jumping", type=["wav"], key="jump")

uploaded_files = {
    "Background": background_file,
    "Heel Drop": heel_drop_file,
    "Walking": walking_file,
    "Running": running_file,
    "Jumping": jumping_file,
}


def read_wav(uploaded_file):
    fs, x = wavfile.read(io.BytesIO(uploaded_file.getvalue()))
    if x.ndim == 1:
        x = x[:, None]
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        x = x.astype(np.float64) / max(abs(info.min), info.max)
    else:
        x = x.astype(np.float64)
    return float(fs), x


def to_acceleration(x):
    if samples_are_acceleration:
        return x
    return x * full_scale_v / ((sensitivity_mv / 1000.0) * chain_gain)


def condition_signal(a, fs):
    y = signal.detrend(a, type="constant")
    if 0 < high_pass_hz < fs / 2:
        sos = signal.butter(4, high_pass_hz, "highpass", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, y)
    if 0 < low_pass_hz < fs / 2:
        sos = signal.butter(4, low_pass_hz, "lowpass", fs=fs, output="sos")
        y = signal.sosfiltfilt(sos, y)
    return y


def peak_spectrum(a, fs):
    n = len(a)
    window = signal.windows.hann(n, sym=False)
    transformed = np.fft.rfft((a - np.mean(a)) * window)
    amplitude = np.abs(transformed) / (n * np.mean(window))
    if len(amplitude) > 2:
        amplitude[1:-1] *= 2
    return np.fft.rfftfreq(n, 1 / fs), amplitude


def welch_psd(a, fs):
    nperseg = min(len(a), max(256, int(fs * 2)))
    f, p = signal.welch(a, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2)
    return f, 10 * np.log10(np.maximum(p, np.finfo(float).tiny))


def extract_peaks(f, db, background):
    mask = (f >= frequency_min_hz) & (f <= min(frequency_max_hz, f[-1]))
    fm, dm = f[mask], db[mask]
    if len(fm) < 3:
        return []
    ids, props = signal.find_peaks(dm, prominence=prominence_db)
    if background is None:
        bg = np.full_like(fm, np.nan)
    else:
        bg = np.interp(fm, background.psd_frequency, background.psd_db, left=np.nan, right=np.nan)
    rows = []
    for j, idx in enumerate(ids):
        rows.append({
            "Frequency_Hz": float(fm[idx]),
            "PSD_dB": float(dm[idx]),
            "Prominence_dB": float(props["prominences"][j]),
            "Above_background_dB": float(dm[idx] - bg[idx]) if np.isfinite(bg[idx]) else np.nan,
        })
    return sorted(rows, key=lambda row: row["PSD_dB"], reverse=True)


def detect_impacts(a, fs):
    envelope = np.abs(signal.hilbert(a))
    n = max(1, int(0.02 * fs))
    smooth = np.convolve(envelope, np.ones(n) / n, mode="same")
    median = np.median(smooth)
    sigma = 1.4826 * np.median(np.abs(smooth - median)) + np.finfo(float).eps
    ids, _ = signal.find_peaks(
        smooth,
        height=median + detection_sigma * sigma,
        distance=max(1, int(minimum_event_gap_s * fs)),
    )
    refined = []
    radius = max(1, int(0.15 * fs))
    for idx in ids:
        lo, hi = max(0, idx - radius), min(len(a), idx + radius)
        refined.append(lo + int(np.argmax(np.abs(a[lo:hi]))))
    return np.asarray(refined, dtype=int)


def make_background(file, monitor=None):
    def bg_report(percent, message):
        if monitor is not None:
            monitor(percent, message)
    if file is None:
        return None
    bg_report(5, "Reading background WAV")
    fs, x = read_wav(file)
    bg_report(30, f"Background loaded: {len(x):,} samples at {fs:g} Hz")
    a = condition_signal(to_acceleration(x[:, 0]), fs)
    bg_report(55, "Background filtering complete")
    pf, pdb = welch_psd(a, fs)
    bg_report(75, "Background PSD complete")
    sf, sa = peak_spectrum(a, fs)
    bg_report(100, "Background analysis complete")
    return BackgroundRecord(
        rms=float(np.sqrt(np.mean(a ** 2))),
        peak=float(np.max(np.abs(a))),
        psd_frequency=pf,
        psd_db=pdb,
        spectrum_frequency=sf,
        spectrum_amplitude=sa,
    )


def analyse_test(test_type, file, background, monitor=None):
    def report(percent, message):
        if monitor is not None:
            monitor(percent, message)
    report(5, "Reading WAV")
    fs, x = read_wav(file)
    report(15, f"Loaded {len(x):,} samples at {fs:g} Hz")
    a = condition_signal(to_acceleration(x[:, 0]), fs)
    report(25, "Filtering and calibration complete")
    t = np.arange(len(a)) / fs
    measured_peak = float(np.max(np.abs(a)))
    measured_rms = float(np.sqrt(np.mean(a ** 2)))
    corrected_rms = (
        math.sqrt(max(0.0, measured_rms ** 2 - background.rms ** 2))
        if background is not None else np.nan
    )
    report(32, "Calculating peak spectrum")
    sf, sa = peak_spectrum(a, fs)
    report(40, "Calculating Welch PSD")
    pf, pdb = welch_psd(a, fs)
    peaks = extract_peaks(pf, pdb, background)
    report(45, f"Found {len(peaks)} spectral peak candidates")
    event_rows, impact_times, event_freqs = [], [], []

    if test_type == "Heel Drop":
        report(48, "Detecting heel-drop impacts")
        impacts = detect_impacts(a, fs)
        impact_times = [float(i / fs) for i in impacts]
        report(52, f"Detected {len(impacts)} heel-drop candidates")
        for number, idx in enumerate(impacts, start=1):
            report(52 + int(43 * number / max(1, len(impacts))), f"Decay analysis {number}/{len(impacts)}")
            start = idx + int(impact_transient_s * fs)
            end = min(
                len(a),
                start + int(decay_window_s * fs),
                idx + int(max(0.5, minimum_event_gap_s - 0.1) * fs),
            )
            if end - start < 128:
                continue
            decay_f, decay_db = welch_psd(a[start:end], fs)
            decay_peaks = extract_peaks(decay_f, decay_db, background)
            natural_frequency = decay_peaks[0]["Frequency_Hz"] if decay_peaks else np.nan
            if np.isfinite(natural_frequency):
                event_freqs.append(natural_frequency)
            lo, hi = max(0, idx - int(0.15 * fs)), min(len(a), idx + int(0.5 * fs))
            event_peak = float(np.max(np.abs(a[lo:hi])))
            decay = a[start:end]
            decay_time = np.arange(len(decay)) / fs
            event_rows.append({
                "Event": number,
                "Time_s": idx / fs,
                "Natural_frequency_Hz": natural_frequency,
                "Peak_acceleration_mm_s2": event_peak * 1000,
                "Above_background_dB": decay_peaks[0]["Above_background_dB"] if decay_peaks else np.nan,
                "Decay_time": decay_time,
                "Decay_acceleration": decay,
            })

    if event_freqs:
        dominant_frequency = float(np.median(event_freqs))
    elif peaks:
        dominant_frequency = peaks[0]["Frequency_Hz"]
    else:
        mask = (pf >= frequency_min_hz) & (pf <= min(frequency_max_hz, pf[-1]))
        dominant_frequency = float(pf[np.where(mask)[0][np.argmax(pdb[mask])]]) if np.any(mask) else np.nan

    report(100, f"{test_type} complete")
    return {
        "type": test_type,
        "file_name": file.name,
        "fs": fs,
        "time": t,
        "acceleration": a,
        "spectrum_frequency": sf,
        "spectrum_amplitude": sa,
        "peak": measured_peak,
        "rms": measured_rms,
        "corrected_rms": corrected_rms,
        "dominant_frequency": dominant_frequency,
        "above_background_db": peaks[0]["Above_background_dB"] if peaks else np.nan,
        "peaks": peaks,
        "impact_times": impact_times,
        "events": event_rows,
    }


def chart_png(result, background):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.5), constrained_layout=True)
    axes[0].plot(result["time"], result["acceleration"], linewidth=0.7)
    for i, impact_time in enumerate(result["impact_times"], start=1):
        axes[0].axvline(impact_time, color="#D71920", alpha=0.45, linewidth=0.8)
        axes[0].text(impact_time, axes[0].get_ylim()[1], f"HD{i}", fontsize=7, rotation=90, va="top")
    axes[0].set_title(f'{result["type"]} - Time History')
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Acceleration [m/s²]")
    axes[0].grid(alpha=0.25)

    f, amp = result["spectrum_frequency"], result["spectrum_amplitude"]
    mask = (f >= frequency_min_hz) & (f <= min(frequency_max_hz, f[-1]))
    axes[1].plot(f[mask], amp[mask], label=result["type"], linewidth=1)
    if background is not None:
        bf, ba = background.spectrum_frequency, background.spectrum_amplitude
        bmask = (bf >= frequency_min_hz) & (bf <= min(frequency_max_hz, bf[-1]))
        axes[1].plot(bf[bmask], ba[bmask], label="Background", linewidth=1, alpha=0.8)
    axes[1].set_title("Peak Amplitude Spectrum")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Peak acceleration amplitude [m/s²]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=160)
    plt.close(fig)
    output.seek(0)
    return output


def summary_dataframe(background, results):
    rows = []
    if background is not None:
        rows.append({
            "Test": "Background",
            "Primary frequency (Hz)": np.nan,
            "Amplitude / level": f"RMS {background.rms * 1000:.3f} mm/s²",
            "Secondary result": f"Peak {background.peak * 1000:.3f} mm/s²",
        })
    for test_type in ["Heel Drop", "Walking", "Running", "Jumping"]:
        if test_type not in results:
            continue
        result = results[test_type]
        label = "Natural frequency" if test_type == "Heel Drop" else "Dominant frequency"
        rows.append({
            "Test": test_type,
            "Primary frequency (Hz)": round(result["dominant_frequency"], 3) if np.isfinite(result["dominant_frequency"]) else np.nan,
            "Amplitude / level": f"Peak {result['peak'] * 1000:.3f} mm/s²",
            "Secondary result": f"{label}; RMS {result['rms'] * 1000:.3f} mm/s²",
        })
    return pd.DataFrame(rows)


def create_excel(summary, results, images, background):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary", startrow=3)
        wb = writer.book
        title_fmt = wb.add_format({"bold": True, "font_size": 18, "font_color": "#FFFFFF", "bg_color": "#D71920", "align": "left"})
        head_fmt = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#17365D", "border": 1, "text_wrap": True, "valign": "vcenter"})
        ws = writer.sheets["Summary"]
        ws.merge_range("A1:D1", "Arup Acoustics | Floor Vibration Test Summary", title_fmt)
        ws.set_row(3, 30, head_fmt)
        ws.set_column("A:A", 18)
        ws.set_column("B:B", 24)
        ws.set_column("C:C", 28)
        ws.set_column("D:D", 42)
        ws.freeze_panes(4, 0)
        for test_type, result in results.items():
            peak_df = pd.DataFrame(result["peaks"][:10])
            event_df = pd.DataFrame([{k: v for k, v in row.items() if k not in ["Decay_time", "Decay_acceleration"]} for row in result["events"]])
            sheet_name = test_type[:31]
            peak_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=3)
            sh = writer.sheets[sheet_name]
            sh.merge_range("A1:E1", f"Arup Acoustics | {test_type}", title_fmt)
            sh.set_row(3, 30, head_fmt)
            sh.set_column("A:E", 22)
            if not event_df.empty:
                event_df.to_excel(writer, index=False, sheet_name=f"{test_type[:20]} Events")
            image = images[test_type]
            image.seek(0)
            sh.insert_image("G3", f"{test_type}.png", {"image_data": image, "x_scale": 0.65, "y_scale": 0.65})
    return output.getvalue()


def draw_pdf_table(axis, dataframe):
    axis.axis("off")
    display = dataframe.copy()
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        colWidths=[0.18, 0.20, 0.27, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#808080")
        cell.get_text().set_wrap(True)
        if row == 0:
            cell.set_facecolor("#17365D")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
    return table


def create_pdf(summary, results, images, background):
    output = io.BytesIO()
    with PdfPages(output) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patches.extend([plt.Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color="#D71920", zorder=-1)])
        fig.text(0.06, 0.955, "Arup Acoustics", color="white", fontsize=20, weight="bold", va="center")
        fig.text(0.06, 0.89, "Floor Vibration Test Summary", fontsize=17, weight="bold", color="#17365D")
        fig.text(0.06, 0.855, "Background-corrected RMS is energy-based. Peak acceleration remains the measured value.", fontsize=8)
        ax = fig.add_axes([0.05, 0.36, 0.90, 0.44])
        draw_pdf_table(ax, summary)
        notes = [
            "Heel Drop: reported frequency is the median free-decay natural-frequency candidate.",
            "Walking / Running / Jumping: reported frequency is the dominant measured response, which may include cadence harmonics.",
            "Spectral confidence should be reviewed against the uploaded background PSD.",
        ]
        fig.text(0.06, 0.28, "Interpretation notes", fontsize=11, weight="bold", color="#17365D")
        for i, note in enumerate(notes):
            fig.text(0.07, 0.245 - 0.035 * i, f"• {note}", fontsize=8.5, wrap=True)
        fig.text(0.06, 0.06, "Arup Acoustics | Floor Vibration Test Analyser", fontsize=8, color="#666666")
        pdf.savefig(fig)
        plt.close(fig)

        for test_type in ["Heel Drop", "Walking", "Running", "Jumping"]:
            if test_type not in results:
                continue
            result = results[test_type]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patches.extend([plt.Rectangle((0, 0.93), 1, 0.07, transform=fig.transFigure, color="#D71920", zorder=-1)])
            fig.text(0.06, 0.955, "Arup Acoustics", color="white", fontsize=18, weight="bold", va="center")
            fig.text(0.06, 0.89, test_type, fontsize=17, weight="bold", color="#17365D")
            freq_label = "Natural frequency" if test_type == "Heel Drop" else "Dominant response frequency"
            fig.text(0.06, 0.845, f"{freq_label}: {result['dominant_frequency']:.2f} Hz", fontsize=10)
            fig.text(0.06, 0.815, f"Peak acceleration: {result['peak'] * 1000:.3f} mm/s²", fontsize=10)
            fig.text(0.50, 0.815, f"Measured RMS: {result['rms'] * 1000:.3f} mm/s²", fontsize=10)
            image = images[test_type]
            image.seek(0)
            image_array = plt.imread(image)
            ax = fig.add_axes([0.06, 0.13, 0.88, 0.64])
            ax.imshow(image_array)
            ax.axis("off")
            fig.text(0.06, 0.06, "Arup Acoustics | Floor Vibration Test Analyser", fontsize=8, color="#666666")
            pdf.savefig(fig)
            plt.close(fig)

            if test_type == "Heel Drop" and result["events"]:
                for event in result["events"]:
                    event_fig = plt.figure(figsize=(8.27, 11.69))
                    event_fig.patches.extend([plt.Rectangle((0, 0.93), 1, 0.07, transform=event_fig.transFigure, color="#D71920", zorder=-1)])
                    event_fig.text(0.06, 0.955, "Arup Acoustics", color="white", fontsize=18, weight="bold", va="center")
                    event_fig.text(0.06, 0.89, f"Heel Drop Event {event['Event']} - Free Decay", fontsize=16, weight="bold", color="#17365D")
                    event_fig.text(0.06, 0.845, f"Natural frequency: {event['Natural_frequency_Hz']:.2f} Hz", fontsize=10)
                    event_fig.text(0.06, 0.815, f"Event peak acceleration: {event['Peak_acceleration_mm_s2']:.3f} mm/s²", fontsize=10)
                    axd = event_fig.add_axes([0.10, 0.46, 0.82, 0.28])
                    axd.plot(event["Decay_time"], event["Decay_acceleration"], linewidth=0.8, color="#17365D")
                    axd.set_title("Free-decay acceleration used for frequency calculation")
                    axd.set_xlabel("Time from decay-window start [s]")
                    axd.set_ylabel("Acceleration [m/s²]")
                    axd.grid(alpha=0.25)
                    event_fig.text(0.06, 0.06, "Arup Acoustics | Floor Vibration Test Analyser", fontsize=8, color="#666666")
                    pdf.savefig(event_fig)
                    plt.close(event_fig)
    return output.getvalue()


if st.button("Start analysis", type="primary", use_container_width=True):
    available = {name: file for name, file in uploaded_files.items() if file is not None}
    if not available:
        st.error("Upload at least one WAV file before starting the analysis.")
    else:
        status = st.status("Analysis running", expanded=True)
        overall = st.progress(0, text="Overall progress: 0%")
        current = st.progress(0, text="Current stage")
        stage = st.empty()
        diagnostic = st.empty()
        logs = []

        def add_log(text):
            logs.append(text)
            diagnostic.code("\n".join(logs[-12:]), language=None)

        try:
            stage.write("**Background analysis**")
            add_log("Starting background analysis")
            def bg_monitor(percent, message):
                current.progress(int(percent), text=message)
                overall.progress(int(percent * 0.10), text=f"Overall progress: {int(percent * 0.10)}%")
                add_log(message)
            background = make_background(background_file, monitor=bg_monitor)

            results = {}
            selected = [t for t in ["Heel Drop", "Walking", "Running", "Jumping"] if uploaded_files[t] is not None]
            for test_no, test_type in enumerate(selected):
                base = 10 + int(65 * test_no / max(1, len(selected)))
                span = 65 / max(1, len(selected))
                stage.write(f"**Current test: {test_type}**")
                add_log(f"Starting {test_type}")
                def test_monitor(percent, message, base=base, span=span, test_type=test_type):
                    current.progress(int(percent), text=f"{test_type}: {message}")
                    op = min(75, int(base + span * percent / 100))
                    overall.progress(op, text=f"Overall progress: {op}%")
                    add_log(f"{test_type}: {message}")
                results[test_type] = analyse_test(test_type, uploaded_files[test_type], background, monitor=test_monitor)

            stage.write("**Preparing charts**")
            current.progress(0, text="Rendering report figures")
            summary = summary_dataframe(background, results)
            images = {}
            for n, (name, result) in enumerate(results.items(), 1):
                add_log(f"Rendering chart {n}/{len(results)}: {name}")
                images[name] = chart_png(result, background)
                current.progress(int(100*n/max(1,len(results))), text=f"Charts {n}/{len(results)}")
                overall.progress(75 + int(10*n/max(1,len(results))), text="Preparing charts")

            stage.write("**Generating Excel report**")
            add_log("Generating Excel report")
            overall.progress(88, text="Generating Excel report")
            excel_bytes = create_excel(summary, results, images, background)

            stage.write("**Generating A4 PDF report**")
            add_log("Generating PDF report, including heel-drop decay pages")
            overall.progress(93, text="Generating A4 PDF report")
            pdf_bytes = create_pdf(summary, results, images, background)

            st.session_state["analysis_output"] = {"background": background, "results": results, "summary": summary, "images": images, "excel": excel_bytes, "pdf": pdf_bytes}
            current.progress(100, text="Complete")
            overall.progress(100, text="Overall progress: 100%")
            add_log("Complete")
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.exception(exc)

if "analysis_output" in st.session_state:
    output = st.session_state["analysis_output"]
    st.markdown("## Summary")
    st.dataframe(output["summary"], use_container_width=True, hide_index=True)
    for test_type in ["Heel Drop", "Walking", "Running", "Jumping"]:
        if test_type not in output["results"]:
            continue
        result = output["results"][test_type]
        st.markdown(f"## {test_type}")
        m1, m2, m3 = st.columns(3)
        frequency_label = "Natural frequency" if test_type == "Heel Drop" else "Dominant frequency"
        m1.metric(frequency_label, f"{result['dominant_frequency']:.2f} Hz")
        m2.metric("Peak acceleration", f"{result['peak'] * 1000:.3f} mm/s²")
        m3.metric("Measured RMS", f"{result['rms'] * 1000:.3f} mm/s²")
        time_fig = go.Figure()
        time_fig.add_trace(go.Scatter(x=result["time"], y=result["acceleration"], name="Test", line={"width": 1}))
        for i, event_time in enumerate(result["impact_times"], start=1):
            time_fig.add_vline(x=event_time, line_width=1, line_dash="dot", line_color="red")
            time_fig.add_annotation(x=event_time, y=1, yref="paper", text=f"HD{i}", showarrow=False, textangle=-90)
        time_fig.update_layout(title=f"{test_type} - Time History", xaxis_title="Time [s]", yaxis_title="Acceleration [m/s²]", height=360)
        st.plotly_chart(time_fig, use_container_width=True)
        spectrum_fig = go.Figure()
        f, a = result["spectrum_frequency"], result["spectrum_amplitude"]
        mask = (f >= frequency_min_hz) & (f <= min(frequency_max_hz, f[-1]))
        spectrum_fig.add_trace(go.Scatter(x=f[mask], y=a[mask], name=test_type))
        if output["background"] is not None:
            bg = output["background"]
            bmask = (bg.spectrum_frequency >= frequency_min_hz) & (bg.spectrum_frequency <= min(frequency_max_hz, bg.spectrum_frequency[-1]))
            spectrum_fig.add_trace(go.Scatter(x=bg.spectrum_frequency[bmask], y=bg.spectrum_amplitude[bmask], name="Background"))
        spectrum_fig.update_layout(title="Peak Amplitude Spectrum", xaxis_title="Frequency [Hz]", yaxis_title="Peak acceleration amplitude [m/s²]", height=360)
        st.plotly_chart(spectrum_fig, use_container_width=True)

        if test_type == "Heel Drop" and result["events"]:
            st.markdown("### Individual heel-drop free-decay curves")
            st.caption("Each chart shows the post-impact free-decay segment used to calculate that event's natural-frequency candidate.")
            for event in result["events"]:
                with st.expander(
                    f"Heel Drop {event['Event']} | {event['Natural_frequency_Hz']:.2f} Hz | {event['Peak_acceleration_mm_s2']:.3f} mm/s²",
                    expanded=False,
                ):
                    decay_fig = go.Figure()
                    decay_fig.add_trace(
                        go.Scatter(
                            x=event["Decay_time"],
                            y=event["Decay_acceleration"],
                            name=f"HD{event['Event']} decay",
                            line={"width": 1},
                        )
                    )
                    decay_fig.update_layout(
                        title=f"Heel Drop {event['Event']} - Free Decay used for frequency calculation",
                        xaxis_title="Time from decay-window start [s]",
                        yaxis_title="Acceleration [m/s²]",
                        height=320,
                    )
                    st.plotly_chart(decay_fig, use_container_width=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download Excel report", output["excel"], "Arup_Acoustics_Floor_Vibration_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with d2:
        st.download_button("Download A4 PDF report", output["pdf"], "Arup_Acoustics_Floor_Vibration_Report.pdf", "application/pdf", use_container_width=True)
