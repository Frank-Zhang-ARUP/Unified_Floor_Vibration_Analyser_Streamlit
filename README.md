# Floor Vibration Test Analyser
Unified Streamlit tool for Background, Heel Drop, Walking, Running, Jumping and Other Activity WAV analysis.

## Background treatment
- No direct time-domain subtraction.
- RMS energy correction: sqrt(test RMS^2 - background RMS^2), clipped at zero.
- Measured peak acceleration remains uncorrected.
- Spectral candidates include dB above background PSD.

## Reports
The app exports an Excel workbook and a PDF report, both containing time-history and peak-spectrum figures.

## Streamlit deployment
Place `app.py`, `requirements.txt`, and `README.md` in the repository root. Main file path: `app.py`.
