# Unified Floor Vibration Test Analyser

A Streamlit tool for:

- Background vibration
- Heel-drop tests
- Walking
- Running
- Jumping
- Other floor activity

## Background treatment

- No direct time-domain subtraction.
- RMS energy correction: `sqrt(test RMS^2 - background RMS^2)`, clipped at zero.
- Peak acceleration remains the measured peak and is not background-subtracted.
- Frequency candidates report dB above the background PSD.

## Outputs

- Time-history acceleration plot
- Peak amplitude spectrum with optional background overlay
- Measured peak acceleration in m/s², mm/s² and %g
- Measured RMS and background-corrected RMS
- Dominant response frequency
- Automatic heel-drop event detection and per-event frequency analysis
- Excel report with embedded charts
- PDF report with tables and charts

## Streamlit Community Cloud

Place these files in the repository root:

- `app.py`
- `requirements.txt`
- `README.md`

Set the Main file path to `app.py`, then reboot the app after committing the files.

For compatibility, select Python 3.12 in Streamlit Advanced settings if available.
