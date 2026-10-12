// @ts-nocheck
import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { submissionsService } from '../api/submissions';

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState('');
  const [status, setStatus] = useState('idle');
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [mode, setMode] = useState('multimodal');
  const [showTextInput, setShowTextInput] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const navigate = useNavigate();

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file && !text) return;
    setStatus('uploading');
    setProcessingStep(0);
    try {
      let res;
      if (file) {
         res = await submissionsService.uploadMedia(file, mode);
      } else {
         res = await submissionsService.uploadText(text);
      }
      setSubmissionId(res._id);
      setStatus('processing');
    } catch (e) {
      console.error(e);
      setStatus('error');
    }
  };

  useEffect(() => {
    if (!submissionId || status !== 'processing') return;
    const interval = setInterval(async () => {
      try {
        const res = await submissionsService.getSubmission(submissionId);
        if (res.status === 'completed') {
           clearInterval(interval);
           navigate(`/report/${submissionId}`);
        } else if (res.status === 'failed') {
           clearInterval(interval);
           setStatus('error');
        }
      } catch (e) {}
    }, 2000);
    return () => clearInterval(interval);
  }, [submissionId, status, navigate]);

  useEffect(() => {
    let stepInterval: NodeJS.Timeout;
    if (status === 'processing' || status === 'uploading') {
      stepInterval = setInterval(() => {
        setProcessingStep(prev => Math.min(prev + 1, 4));
      }, 1500);
    }
    return () => clearInterval(stepInterval);
  }, [status]);

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col w-full">
  <div className="relative w-full overflow-hidden">
    {/* Ambient Luminous Waveforms and Forensic Mesh Backdrop */}
    <div className="absolute inset-0 pointer-events-none opacity-40 overflow-hidden">
      <svg className="w-full h-full" fill="none" viewBox="0 0 1440 800" xmlns="http://www.w3.org/2000/svg">
        <path className="text-cyan-energy" d="M-100 200 C300 120, 600 420, 1100 220 C1300 140, 1500 280, 1600 240" stroke="currentColor" strokeDasharray="6 6" strokeWidth="1.5" />
        <path className="text-primary-container" d="M-50 320 C400 180, 750 480, 1200 280 C1380 200, 1550 340, 1650 300" opacity="0.3" stroke="currentColor" strokeWidth={2} />
        <circle className="text-surface-tint-ice" cx={1180} cy={240} fill="currentColor" opacity="0.5" r={160} />
        <circle className="text-secondary-fixed" cx={280} cy={180} fill="currentColor" opacity="0.25" r={90} />
      </svg>
    </div>
    <div className="relative max-w-[1280px] w-full mx-auto px-margin py-space-xl flex flex-col gap-space-xl">
      {/* Studio Header & Real-Time Operational Telemetry */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-space-lg">
        <div className="flex flex-col gap-space-xs max-w-2xl">
          <div className="flex items-center gap-space-xs">
            <span className="px-space-sm py-space-xs rounded-full bg-surface-tint-ice text-primary font-label-code text-label-code uppercase tracking-wider">
              FORENSIC INGESTION PROTOCOL v4.2
            </span>
            <span className="w-2 h-2 rounded-full bg-cyan-energy animate-ping" />
          </div>
          <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight">
            ANALYZE CONTENT STUDIO
          </h1>
          <p className="font-body-lg text-body-lg text-slate-body">
            Upload suspicious digital assets across any modality. CheckReality AI coordinates deep neural classifiers, weighted ensemble fusion, and explainable AI pipelines in real time.
          </p>
        </div>

      </div>
      {/* Main Studio Workspace */}
      <div className="max-w-4xl mx-auto w-full flex flex-col gap-space-lg">
        {/* Modality Selector Segmented Bar */}
        <div className="bg-surface-crisp shadow-sm rounded-2xl p-space-xs flex flex-wrap gap-space-xs" id="modalitySelector">
          <button onClick={() => setMode('multimodal')} className={`modality-btn flex-1 min-w-[140px] flex items-center justify-center gap-space-xs py-space-sm px-space-md rounded-xl font-label-lg text-label-lg transition-all ${mode === 'multimodal' ? 'bg-primary-container text-on-primary shadow-md' : 'text-on-surface-variant hover:bg-surface-tint-ice'}`} data-mode="multimodal" type="button">
            <span className="material-symbols-outlined text-[20px]">hub</span>
            <div className="flex flex-col items-start leading-tight">
              <span className="font-bold flex items-center gap-1">Multimodal <span className="text-[10px] bg-cyan-energy text-on-surface px-1.5 py-0.5 rounded-full font-label-caption uppercase">Fusion</span></span>
              <span className="text-[10px] opacity-80">Full 4-Way Analysis</span>
            </div>
          </button>
          <button onClick={() => setMode('image')} className={`modality-btn flex-1 min-w-[120px] flex items-center justify-center gap-space-xs py-space-sm px-space-md rounded-xl font-label-lg text-label-lg transition-all ${mode === 'image' ? 'bg-primary-container text-on-primary shadow-md' : 'text-on-surface-variant hover:bg-surface-tint-ice'}`} data-mode="image" type="button">
            <span className="material-symbols-outlined text-[20px] text-primary">image</span>
            <div className="flex flex-col items-start leading-tight">
              <span className="font-semibold">Image</span>
              <span className="text-[10px] text-slate-body">Facial Portraits, .png</span>
            </div>
          </button>
          <button onClick={() => setMode('video')} className={`modality-btn flex-1 min-w-[120px] flex items-center justify-center gap-space-xs py-space-sm px-space-md rounded-xl font-label-lg text-label-lg transition-all ${mode === 'video' ? 'bg-primary-container text-on-primary shadow-md' : 'text-on-surface-variant hover:bg-surface-tint-ice'}`} data-mode="video" type="button">
            <span className="material-symbols-outlined text-[20px] text-secondary">movie</span>
            <div className="flex flex-col items-start leading-tight">
              <span className="font-semibold">Video</span>
              <span className="text-[10px] text-slate-body">Deepfakes, .mp4, .mov</span>
            </div>
          </button>
          <button onClick={() => setMode('audio')} className={`modality-btn flex-1 min-w-[120px] flex items-center justify-center gap-space-xs py-space-sm px-space-md rounded-xl font-label-lg text-label-lg transition-all ${mode === 'audio' ? 'bg-primary-container text-on-primary shadow-md' : 'text-on-surface-variant hover:bg-surface-tint-ice'}`} data-mode="audio" type="button">
            <span className="material-symbols-outlined text-[20px] text-secondary-container">graphic_eq</span>
            <div className="flex flex-col items-start leading-tight">
              <span className="font-semibold">Audio</span>
              <span className="text-[10px] text-slate-body">Voice Clones, .wav</span>
            </div>
          </button>
          <button onClick={() => setMode('text')} className={`modality-btn flex-1 min-w-[120px] flex items-center justify-center gap-space-xs py-space-sm px-space-md rounded-xl font-label-lg text-label-lg transition-all ${mode === 'text' ? 'bg-primary-container text-on-primary shadow-md' : 'text-on-surface-variant hover:bg-surface-tint-ice'}`} data-mode="text" type="button">
            <span className="material-symbols-outlined text-[20px] text-primary">article</span>
            <div className="flex flex-col items-start leading-tight">
              <span className="font-semibold">Text</span>
              <span className="text-[10px] text-slate-body">Articles, GPT Transcripts</span>
            </div>
          </button>
        </div>
        {/* Interactive File Dropzone Area */}
        <div className="relative group bg-surface-crisp shadow-md rounded-3xl p-space-xl transition-all duration-3xl overflow-hidden" id="dropzoneContainer">
          {/* Animated Ambient Glow Spotlights */}
          <div className="absolute -top-24 -right-24 w-72 h-72 bg-cyan-border-glow rounded-full blur-3xl opacity-40 group-hover:opacity-70 transition-opacity" />
          <div className="absolute -bottom-24 -left-24 w-72 h-72 bg-surface-container rounded-full blur-3xl opacity-60" />
          <div className="relative flex flex-col items-center justify-center text-center p-space-2xl rounded-2xl bg-surface-tint-card/80 backdrop-blur-sm cursor-pointer hover:bg-surface-tint-ice/50 transition-all">
            {/* Verification Shield Illustration with Neural Topology (Inspired by Brand Artifact) */}
            <div className="relative w-32 h-32 mb-space-lg flex items-center justify-center">
              {/* Outer Pulsing Halo */}
              <div className="absolute inset-0 rounded-full bg-cyan-energy/20 animate-ping opacity-60" />
              {/* Stylized SVG Forensic Dual-Hemisphere Shield */}
              <img alt="CheckReality AI Logo" className="w-28 h-28 object-contain drop-shadow-xl" src="/logo.png" />
              {/* Floating Micro Indicator Dots */}
              <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-verdict-real flex items-center justify-center shadow-md">
                <span className="material-symbols-outlined text-on-primary text-[12px]">check</span>
              </div>
            </div>
            {/* Main Drop Action Callout */}
            <h3 className="font-headline-sm text-headline-sm text-on-surface font-bold mb-space-xs">
              {mode === 'image' ? 'Drag & drop suspicious image here' : mode === 'video' ? 'Drag & drop suspicious video here' : mode === 'audio' ? 'Drag & drop suspicious audio here' : mode === 'text' ? 'Drag & drop text document here' : 'Drag & drop digital evidence files here'}
            </h3>
            <p className="font-body-md text-body-md text-slate-body max-w-md mb-space-lg">
              {mode === 'image' ? 'Drop JPG, PNG, WEBP.' : mode === 'video' ? 'Drop MP4, MOV.' : mode === 'audio' ? 'Drop WAV, MP3.' : mode === 'text' ? 'Drop TXT, PDF or paste direct text below.' : 'Drop MP4, WAV, JPG, PNG, WEBP, or batch records.'}
            </p>
            {/* Upload CTA Trigger & File Browser Button */}
            <div className="flex flex-wrap items-center justify-center gap-space-md">
              <label className="px-space-xl py-space-sm rounded-full bg-primary text-on-primary font-label-lg text-label-lg font-semibold hover:bg-electric-blue-hover shadow-[0_4px_16px_rgba(0,102,255,0.25)] transition-all cursor-pointer flex items-center gap-space-xs">
                <span className="material-symbols-outlined text-[18px]">cloud_upload</span>
                <span>{file ? file.name : 'Browse Device Files'}</span>
                <input 
                  accept={mode === 'image' ? 'image/*' : mode === 'video' ? 'video/*' : mode === 'audio' ? 'audio/*' : mode === 'text' ? '.txt,.pdf' : 'video/*,audio/*,image/*,.txt'} 
                  className="hidden" 
                  id="primaryFileInput" 
                  type="file" 
                  onChange={(e) => e.target.files && setFile(e.target.files[0])} 
                />
              </label>
              {(mode === 'text' || mode === 'multimodal') && (
                <button onClick={() => setShowTextInput(!showTextInput)} className="px-space-lg py-space-sm rounded-full bg-surface-crisp text-on-surface font-label-lg text-label-lg font-medium shadow-sm hover:bg-surface-tint-ice transition-all flex items-center gap-space-xs" id="toggleDirectTextInput" type="button">
                  <span className="material-symbols-outlined text-[18px] text-primary">edit_note</span>
                  <span>Paste Direct Text</span>
                </button>
              )}
            </div>
          </div>
          
          {/* Direct Text Editor Panel (Collapsible / Toggleable) */}
          <div className={`${(showTextInput && (mode === 'text' || mode === 'multimodal')) ? 'flex' : 'hidden'} mt-space-lg flex-col gap-space-sm bg-surface-tint-ice p-space-lg rounded-2xl`} id="textInputArea">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-space-xs">
                <span className="material-symbols-outlined text-primary text-[20px]">article</span>
                <span className="font-headline-sm text-body-lg text-on-surface font-bold">Synthesizer &amp; LLM Attribution Parser</span>
              </div>
            </div>
            <textarea value={text} onChange={(e) => setText(e.target.value)} className="w-full bg-surface-crisp text-on-surface p-space-md rounded-xl font-body-md text-body-md focus:outline-none focus:ring-2 focus:ring-primary shadow-inner" id="rawTextContent" placeholder="Paste article draft, transcript snippet, or synthetic social statement..." rows={5} />
          </div>
          
          {/* Active Selected Ingestion Staging Item */}
          {file && (
            <div className="mt-space-lg bg-surface-tint-ice rounded-2xl p-space-md flex items-center justify-between gap-space-md" id="stagedFileBox">
              <div className="flex items-center gap-space-md min-w-0">
                <div className="w-12 h-12 rounded-xl bg-primary-container text-on-primary flex items-center justify-center shrink-0 shadow-md">
                  <span className="material-symbols-outlined text-[24px]">
                    {mode === 'image' ? 'image' : mode === 'video' ? 'movie' : mode === 'audio' ? 'graphic_eq' : 'insert_drive_file'}
                  </span>
                </div>
                <div className="flex flex-col min-w-0">
                  <div className="flex items-center gap-space-xs">
                    <span className="font-label-lg text-label-lg text-on-surface font-bold truncate">{file.name}</span>
                  </div>
                  <span className="font-label-code text-body-sm text-slate-body">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB • Ready for Analysis
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-space-sm shrink-0">
                <span className="font-label-code text-body-sm text-verdict-real font-bold flex items-center gap-1">
                  <span className="material-symbols-outlined text-[16px]">check_circle</span> Ready
                </span>
                <button onClick={() => setFile(null)} className="p-space-xs rounded-full text-slate-body hover:bg-surface-crisp hover:text-error transition-all" title="Remove File" type="button">
                  <span className="material-symbols-outlined text-[20px]">close</span>
                </button>
              </div>
            </div>
          )}
          
          {/* Processing UI or Execution Trigger */}
          <div className="mt-space-xl flex justify-center w-full">
            {status === 'idle' || status === 'error' ? (
              <div className="flex flex-col items-center gap-space-sm w-full max-w-[420px]">
                {status === 'error' && <span className="text-error font-bold font-label-code text-sm bg-error/10 px-4 py-1 rounded-full">Analysis Failed. Please try again.</span>}
                <button type="button" disabled={(!file && !text)} onClick={handleUpload} className="group relative w-full py-space-lg px-space-xl rounded-full bg-primary-container text-on-primary font-label-lg text-[20px] font-bold text-center hover:bg-electric-blue-hover shadow-[0_8px_24px_rgba(0,102,255,0.35)] hover:shadow-[0_12px_32px_rgba(0,216,246,0.4)] transition-all flex items-center justify-center gap-space-md overflow-hidden disabled:opacity-50" id="executeAnalysisBtn">
                  <div className="absolute inset-0 w-1/2 h-full bg-white/20 skew-x-12 -translate-x-full group-hover:translate-x-[300%] transition-transform duration-1000" />
                  <span className="material-symbols-outlined text-[28px]">analytics</span>
                  <span className="whitespace-nowrap">EXECUTE DEEPFAKE ANALYSIS</span>
                </button>
              </div>
            ) : (
              <div className="w-full max-w-xl bg-surface-crisp p-space-xl rounded-3xl shadow-[0_12px_40px_rgba(0,0,0,0.08)] border border-outline-variant/30 flex flex-col gap-space-md relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-primary/5 via-cyan-energy/10 to-primary/5 opacity-50 pointer-events-none" />
                <div className="relative flex justify-between items-center mb-space-sm">
                  <span className="font-headline-sm text-on-surface font-bold flex items-center gap-3">
                    <span className="material-symbols-outlined animate-spin text-primary text-[28px]">autorenew</span>
                    Processing Evidence
                  </span>
                  <span className="font-label-code text-primary font-bold text-lg">{Math.min(processingStep * 20 + 17, 99)}%</span>
                </div>
                <div className="relative w-full bg-surface-container rounded-full h-3 overflow-hidden shadow-inner">
                  <div className="bg-gradient-to-r from-primary to-cyan-energy h-full transition-all duration-1000 ease-out relative" style={{ width: `${Math.min(processingStep * 20 + 17, 99)}%` }}>
                    <div className="absolute inset-0 bg-white/20 animate-pulse" />
                  </div>
                </div>
                <div className="relative flex flex-col gap-3 mt-space-sm font-label-code text-body-sm text-slate-body">
                  <div className={`flex items-center gap-3 transition-colors duration-300 ${processingStep >= 0 ? 'text-on-surface font-semibold' : ''}`}><span className={`material-symbols-outlined text-[20px] ${processingStep > 0 ? 'text-verdict-real' : processingStep === 0 ? 'text-primary animate-pulse' : ''}`}>{processingStep > 0 ? 'check_circle' : processingStep === 0 ? 'radio_button_checked' : 'radio_button_unchecked'}</span> Initializing Neural Engines...</div>
                  <div className={`flex items-center gap-3 transition-colors duration-300 ${processingStep >= 1 ? 'text-on-surface font-semibold' : ''}`}><span className={`material-symbols-outlined text-[20px] ${processingStep > 1 ? 'text-verdict-real' : processingStep === 1 ? 'text-primary animate-pulse' : ''}`}>{processingStep > 1 ? 'check_circle' : processingStep === 1 ? 'radio_button_checked' : 'radio_button_unchecked'}</span> Extracting Multi-Modal Features...</div>
                  <div className={`flex items-center gap-3 transition-colors duration-300 ${processingStep >= 2 ? 'text-on-surface font-semibold' : ''}`}><span className={`material-symbols-outlined text-[20px] ${processingStep > 2 ? 'text-verdict-real' : processingStep === 2 ? 'text-primary animate-pulse' : ''}`}>{processingStep > 2 ? 'check_circle' : processingStep === 2 ? 'radio_button_checked' : 'radio_button_unchecked'}</span> Running Inference Ensembles...</div>
                  <div className={`flex items-center gap-3 transition-colors duration-300 ${processingStep >= 3 ? 'text-on-surface font-semibold' : ''}`}><span className={`material-symbols-outlined text-[20px] ${processingStep > 3 ? 'text-verdict-real' : processingStep === 3 ? 'text-primary animate-pulse' : ''}`}>{processingStep > 3 ? 'check_circle' : processingStep === 3 ? 'radio_button_checked' : 'radio_button_unchecked'}</span> Calibrating Confidence Scores...</div>
                  <div className={`flex items-center gap-3 transition-colors duration-300 ${processingStep >= 4 ? 'text-on-surface font-semibold' : ''}`}><span className={`material-symbols-outlined text-[20px] ${processingStep > 4 ? 'text-verdict-real' : processingStep === 4 ? 'text-primary animate-pulse' : ''}`}>{processingStep > 4 ? 'check_circle' : processingStep === 4 ? 'radio_button_checked' : 'radio_button_unchecked'}</span> Finalizing Verdict...</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
    </div>
  );
}
