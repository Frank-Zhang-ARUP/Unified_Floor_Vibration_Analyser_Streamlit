import io, tempfile, math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile
from scipy import signal
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title='Floor Vibration Test Analyser',layout='wide')
st.title('Floor Vibration Test Analyser')
st.caption('Background + Heel Drop + Walking / Running / Jumping analysis, with Excel and PDF reports.')

with st.sidebar:
    mode=st.selectbox('Analysis mode',['Heel Drop','Walking','Running','Jumping','Other Activity'])
    st.header('Calibration')
    sens=st.number_input('Sensor sensitivity [mV/(m/s²)]',0.001,10000.0,10.2,0.1)
    gain=st.number_input('Recording-chain gain',0.000001,1e6,1.0,0.1)
    vfs=st.number_input('WAV full-scale [Vpeak]',0.000001,1000.0,1.0,0.1)
    eng=st.checkbox('WAV samples already represent m/s²',False)
    st.header('Processing')
    hp=st.number_input('High-pass [Hz], 0=off',0.0,100.0,0.5,0.1)
    lp=st.number_input('Low-pass [Hz], 0=off',0.0,5000.0,100.0,1.0)
    fmin=st.number_input('Frequency range min [Hz]',0.1,100.0,0.5,0.1)
    fmax=st.number_input('Frequency range max [Hz]',1.0,1000.0,80.0,1.0)
    prominence=st.number_input('Peak prominence [dB]',1.0,40.0,5.0,0.5)
    st.header('Heel Drop')
    threshold_sigma=st.number_input('Detection threshold [robust sigma]',2.0,30.0,7.0,0.5)
    min_gap=st.number_input('Minimum event gap [s]',0.3,20.0,2.0,0.1)
    decay_delay=st.number_input('Exclude impact transient [s]',0.0,1.0,0.08,0.01)
    decay_time=st.number_input('Decay window [s]',0.5,20.0,4.0,0.25)
    cluster_tol=st.number_input('Frequency consistency tolerance [Hz]',0.1,5.0,0.5,0.1)

bg_file=st.file_uploader('Background WAV (optional, same measurement chain)',type=['wav'],key='bg')
test_files=st.file_uploader('Test WAV file(s)',type=['wav'],accept_multiple_files=True,key='test')

def read_wav(raw):
    fs,x=wavfile.read(io.BytesIO(raw)); x=x[:,None] if x.ndim==1 else x
    if np.issubdtype(x.dtype,np.integer):
        ii=np.iinfo(x.dtype); x=x.astype(float)/max(abs(ii.min),ii.max)
    else:x=x.astype(float)
    return fs,x

def to_acc(x): return x if eng else x*vfs/(sens/1000*gain)
def condition(a,fs):
    a=signal.detrend(a,type='constant')
    if 0<hp<fs/2:a=signal.sosfiltfilt(signal.butter(4,hp,'highpass',fs=fs,output='sos'),a)
    if 0<lp<fs/2:a=signal.sosfiltfilt(signal.butter(4,lp,'lowpass',fs=fs,output='sos'),a)
    return a
def robust_sigma(y):return 1.4826*np.median(np.abs(y-np.median(y)))+np.finfo(float).eps
def peak_spectrum(a,fs):
    n=len(a);w=signal.windows.hann(n,sym=False);y=np.fft.rfft((a-a.mean())*w);amp=np.abs(y)/(n*w.mean())
    if len(amp)>2:amp[1:-1]*=2
    return np.fft.rfftfreq(n,1/fs),amp
def psd(a,fs):
    n=min(len(a),max(256,int(fs*2)));f,p=signal.welch(a,fs=fs,nperseg=n,noverlap=n//2)
    return f,10*np.log10(np.maximum(p,np.finfo(float).tiny))
def interp_bg(f,bgf,bgdb):return np.interp(f,bgf,bgdb,left=np.nan,right=np.nan) if bgf is not None else np.full_like(f,np.nan)
def find_peaks_db(f,db,bgf=None,bgdb=None):
    m=(f>=fmin)&(f<=min(fmax,f[-1]));fm,dm=f[m],db[m]
    ids,pr=signal.find_peaks(dm,prominence=prominence) if len(fm)>2 else ([],{'prominences':[]})
    b=interp_bg(fm,bgf,bgdb);out=[]
    for j,i in enumerate(ids):out.append((float(fm[i]),float(dm[i]),float(pr['prominences'][j]),float(dm[i]-b[i]) if np.isfinite(b[i]) else np.nan))
    return sorted(out,key=lambda z:z[1],reverse=True)
def plots_png(title,t,a,f,amp,bgf=None,bgamp=None,markers=None):
    fig,ax=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    ax[0].plot(t,a,lw=.7);ax[0].set(title=title+' - Time History',xlabel='Time [s]',ylabel='Acceleration [m/s²]');ax[0].grid(alpha=.25)
    if markers:
        for x in markers:ax[0].axvline(x,color='r',alpha=.35,lw=.8)
    m=(f>=fmin)&(f<=min(fmax,f[-1]));ax[1].plot(f[m],amp[m],label='Test',lw=1)
    if bgf is not None and bgamp is not None:ax[1].plot(bgf,bgamp,label='Background',lw=1,alpha=.8)
    ax[1].set(title='Peak Amplitude Spectrum',xlabel='Frequency [Hz]',ylabel='Peak acceleration [m/s²]');ax[1].grid(alpha=.25);ax[1].legend()
    b=io.BytesIO();fig.savefig(b,format='png',dpi=160);plt.close(fig);b.seek(0);return b

def excel_report(summary,events,figures):
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine='xlsxwriter') as w:
        summary.to_excel(w,index=False,sheet_name='Summary');wb=w.book;ws=w.sheets['Summary']
        fmt=wb.add_format({'bold':True,'bg_color':'#17365D','font_color':'white','border':1});ws.set_row(0,24,fmt);ws.freeze_panes(1,0);ws.set_column(0,len(summary.columns)-1,18)
        if len(events):events.to_excel(w,index=False,sheet_name='Heel Drop Events');w.sheets['Heel Drop Events'].set_column(0,len(events.columns)-1,18)
        for i,(name,img) in enumerate(figures[:20],1):
            sh=wb.add_worksheet(('Chart_%02d'%i));sh.write('A1',name,fmt);sh.insert_image('A3',name+'.png',{'image_data':img,'x_scale':.72,'y_scale':.72})
    return out.getvalue()
def pdf_report(summary,events,figures):
    out=io.BytesIO()
    with PdfPages(out) as pdf:
        fig=plt.figure(figsize=(11.69,8.27));fig.text(.06,.92,'Floor Vibration Test Report',fontsize=20,weight='bold');fig.text(.06,.86,f'Analysis mode: {mode}',fontsize=12);fig.text(.06,.82,f'Background correction: {"PSD comparison / energy correction" if bg_file else "No background file"}',fontsize=11)
        ax=fig.add_axes([.05,.08,.9,.66]);ax.axis('off');show=summary.copy().round(5);tbl=ax.table(cellText=show.values,colLabels=show.columns,loc='upper left',cellLoc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(6);tbl.scale(1,1.35);pdf.savefig(fig);plt.close(fig)
        for name,img in figures:
            img.seek(0);arr=plt.imread(img);fig=plt.figure(figsize=(11.69,8.27));plt.imshow(arr);plt.axis('off');plt.title(name);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
        if len(events):
            fig=plt.figure(figsize=(11.69,8.27));ax=fig.add_axes([.04,.06,.92,.86]);ax.axis('off');show=events.copy().round(5);tbl=ax.table(cellText=show.values,colLabels=show.columns,loc='upper left',cellLoc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(6);tbl.scale(1,1.25);plt.title('Heel Drop Event Results');pdf.savefig(fig);plt.close(fig)
    return out.getvalue()

bgf=bgdb=bgaf=bgamp=None;bg_rms=bg_peak=np.nan
if bg_file:
    bfs,bx=read_wav(bg_file.getvalue());ba=condition(to_acc(bx[:,0]),bfs);bgf,bgdb=psd(ba,bfs);bgaf,bgamp=peak_spectrum(ba,bfs);bg_rms=float(np.sqrt(np.mean(ba**2)));bg_peak=float(np.max(np.abs(ba)))
    st.success(f'Background loaded: RMS {bg_rms*1000:.3f} mm/s²; peak {bg_peak*1000:.3f} mm/s²')

if not test_files:st.info('Upload test WAV files. Use a background WAV recorded with the same instrument settings and sensor location where possible.')
else:
    summaries=[];events=[];figures=[]
    for uf in test_files:
      fs,x=read_wav(uf.getvalue())
      for ch in range(x.shape[1]):
        a=condition(to_acc(x[:,ch]),fs);t=np.arange(len(a))/fs;name=f'{uf.name} Ch{ch+1}';peak=float(np.max(np.abs(a)));rms=float(np.sqrt(np.mean(a*a)));corrected_rms=math.sqrt(max(0,rms*rms-bg_rms*bg_rms)) if np.isfinite(bg_rms) else np.nan
        f,amp=peak_spectrum(a,fs);pf,pdb=psd(a,fs);peaks=find_peaks_db(pf,pdb,bgf,bgdb);dom=peaks[0][0] if peaks else float(pf[np.argmax(pdb)])
        impact_times=[];valid_event_freq=[]
        if mode=='Heel Drop':
            env=np.abs(signal.hilbert(a));n=max(1,int(.02*fs));sm=np.convolve(env,np.ones(n)/n,'same');thr=np.median(sm)+threshold_sigma*robust_sigma(sm);ids,_=signal.find_peaks(sm,height=thr,distance=max(1,int(min_gap*fs)))
            impact_times=[i/fs for i in ids]
            for j,p in enumerate(ids,1):
                ds=p+int(decay_delay*fs);de=min(len(a),ds+int(decay_time*fs),p+int(max(.5,min_gap-.1)*fs))
                if de-ds<128:continue
                seg=a[ds:de];ef,edb=psd(seg,fs);ep=find_peaks_db(ef,edb,bgf,bgdb);efn=ep[0][0] if ep else np.nan
                events.append({'File/channel':name,'Event':j,'Time_s':p/fs,'Peak_acc_m_s2':float(np.max(np.abs(a[max(0,p-int(.15*fs)):min(len(a),p+int(.5*fs))]))),'Dominant_Hz':efn,'Above_background_dB':ep[0][3] if ep else np.nan});
                if np.isfinite(efn):valid_event_freq.append(efn)
            if valid_event_freq:dom=float(np.median(valid_event_freq))
        summaries.append({'Mode':mode,'File/channel':name,'Duration_s':len(a)/fs,'Peak_m_s2':peak,'Peak_mm_s2':peak*1000,'Peak_pct_g':peak/9.80665*100,'RMS_m_s2':rms,'Background_RMS_m_s2':bg_rms,'Background_corrected_RMS_m_s2':corrected_rms,'Dominant_response_Hz':dom,'Dominant_above_background_dB':peaks[0][3] if peaks else np.nan,'Heel_drop_count':len(valid_event_freq) if mode=='Heel Drop' else np.nan})
        bgplot=None
        if bgaf is not None:
            mm=(bgaf>=fmin)&(bgaf<=min(fmax,bgaf[-1]));bgplot=(bgaf[mm],bgamp[mm])
        img=plots_png(name,t,a,f,amp,bgplot[0] if bgplot else None,bgplot[1] if bgplot else None,impact_times);figures.append((name,img))
        st.header(name);c1,c2,c3,c4=st.columns(4);c1.metric('Peak acceleration',f'{peak*1000:.3f} mm/s²');c2.metric('Measured RMS',f'{rms*1000:.3f} mm/s²');c3.metric('Background-corrected RMS',f'{corrected_rms*1000:.3f} mm/s²' if np.isfinite(corrected_rms) else 'N/A');c4.metric('Dominant response',f'{dom:.2f} Hz')
# ---- Time history plot ----
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=t,
        y=a,
        name="Test"
    )
)

for z in impact_times:
    fig.add_vline(
        x=z,
        line_width=1,
        line_dash="dot",
        line_color="red"
    )

fig.update_layout(
    xaxis_title="Time [s]",
    yaxis_title="Acceleration [m/s²]",
    height=350
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---- Peak amplitude spectrum ----
m = (f >= fmin) & (f <= min(fmax, f[-1]))

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=f[m],
        y=amp[m],
        name="Test"
    )
)

if bgplot:
    fig.add_trace(
        go.Scatter(
            x=bgplot[0],
            y=bgplot[1],
            name="Background"
        )
    )

fig.update_layout(
    xaxis_title="Frequency [Hz]",
    yaxis_title="Peak acceleration [m/s²]",
    height=350
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.dataframe(
    pd.DataFrame(
        peaks[:10],
        columns=[
            "Frequency_Hz",
            "PSD_dB",
            "Prominence_dB",
            "Above_background_dB"
        ]
    ),
    use_container_width=True
)
          m=(f>=fmin)&(f<=min(fmax,f[-1]));fig=go.Figure(go.Scatter(x=f[m],y=amp[m],name='Test'))
        if bgplot:fig.add_trace(go.Scatter(x=bgplot[0],y=bgplot[1],name='Background'))
        fig.update_layout(xaxis_title='Frequency [Hz]',yaxis_title='Peak acceleration [m/s²]',height=350);st.plotly_chart(fig,use_container_width=True)
        st.dataframe(pd.DataFrame(peaks[:10],columns=['Frequency_Hz','PSD_dB','Prominence_dB','Above_background_dB']),use_container_width=True)
    summary=pd.DataFrame(summaries);eventdf=pd.DataFrame(events)
    st.header('Combined results');st.dataframe(summary,use_container_width=True)
    if len(eventdf):st.subheader('Heel-drop events');st.dataframe(eventdf,use_container_width=True)
    st.download_button('Download Excel report with charts',excel_report(summary,eventdf,figures),'floor_vibration_report.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.download_button('Download PDF report with charts',pdf_report(summary,eventdf,figures),'floor_vibration_report.pdf','application/pdf')
    st.warning('Background correction is applied only to RMS energy: sqrt(total RMS² - background RMS²). Peak acceleration is reported as measured and is not arithmetically background-subtracted. Spectral peaks are assessed by level above the background PSD.')
