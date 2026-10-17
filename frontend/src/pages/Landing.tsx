// @ts-nocheck
import { Link } from 'react-router-dom';

export default function Landing() {

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col w-full">
  {/* HERO SECTION */}
  <section className="relative w-full overflow-hidden bg-surface-bright py-space-2xl lg:py-space-3xl">
    {/* Ethereal Cyber Circuit Ambient Glow Background */}
    <div className="absolute inset-0 pointer-events-none overflow-hidden"><div className="absolute -top-36 left-1/2 -translate-x-1/2 w-[980px] h-[580px] bg-gradient-to-b from-cyan-border-glow/40 via-surface-tint-ice/50 to-transparent rounded-full blur-3xl opacity-70" /><div className="absolute top-1/4 -right-48 w-96 h-96 bg-cyan-energy/10 rounded-full blur-3xl" /><div className="absolute top-2/3 -left-48 w-96 h-96 bg-primary/10 rounded-full blur-3xl" /><svg className="absolute inset-0 w-full h-full opacity-35" height="100%" width="100%" xmlns="http://www.w3.org/2000/svg"><defs><pattern height={28} id="dotPattern" patternUnits="userSpaceOnUse" width={28}><circle cx={2} cy={2} fill="#0050cb" opacity="0.18" r="1.2" /></pattern><linearGradient id="flowWave" x1="0%" x2="100%" y1="0%" y2="100%"><stop offset="0%" stopColor="#0050cb" stopOpacity="0.12" /><stop offset="50%" stopColor="#00D8F6" stopOpacity="0.2" /><stop offset="100%" stopColor="#0066ff" stopOpacity="0.05" /></linearGradient></defs><rect fill="url(#dotPattern)" height="100%" width="100%" /><path d="M0,280 C320,180 460,420 800,240 C1140,60 1280,340 1600,220 L1600,600 L0,600 Z" fill="url(#flowWave)" opacity="0.7" /><path d="M0,320 C240,420 540,200 900,320 C1260,440 1440,260 1600,300 L1600,600 L0,600 Z" fill="url(#flowWave)" opacity="0.4" /></svg></div>
    <div className="relative w-full max-w-[1280px] mx-auto px-margin flex flex-col items-center text-center z-10">
      {/* Academic & Enterprise IEEE 830 Badge */}
      <div className="inline-flex items-center gap-space-xs px-space-md py-space-xs rounded-full bg-surface-tint-ice shadow-sm mb-space-lg">
        <span className="w-2 h-2 rounded-full bg-cyan-energy animate-ping" />
        <span className="font-label-code text-label-code text-secondary font-semibold uppercase tracking-wider">
          ACADEMIC &amp; ENTERPRISE GRADE MULTIMODAL VERIFICATION • IEEE 830 SPEC
        </span>
      </div>
      {/* High Impact Headline */}
      <h1 className="font-headline-hero text-headline-hero text-on-surface tracking-tight max-w-4xl mx-auto mb-space-md">
        VERIFY. DETECT. <span className="bg-gradient-to-r from-primary via-primary-container to-secondary-container bg-clip-text text-transparent">STAY REAL.</span>
      </h1>
      {/* Subheading */}
      <p className="font-body-lg text-body-lg text-slate-body max-w-2xl mx-auto mb-space-xl">
        The multimodal AI intelligence platform engineered to expose deepfakes, synthetic media, and manipulated news across Text, Image, Video, and Audio in real-time.
      </p>
      {/* Primary & Secondary CTA Buttons */}
      <div className="flex flex-wrap items-center justify-center gap-space-md mb-space-2xl">
        <a className="inline-flex items-center gap-space-sm px-space-xl py-3 rounded-full bg-primary-container text-on-primary font-label-lg text-label-lg shadow-[0_4px_20px_rgba(0,102,255,0.35)] hover:bg-electric-blue-hover transition-all group" href="#demo-inspector">
          <span className>ANALYZE CONTENT NOW</span>
          <span className="material-symbols-outlined text-[20px] transition-transform group-hover:translate-x-1">arrow_forward</span>
        </a>
        <a className="inline-flex items-center gap-space-sm px-space-lg py-3 rounded-full bg-surface-crisp text-on-surface font-label-lg text-label-lg shadow-sm hover:bg-surface-tint-ice transition-all" href="#pipeline-architecture">
          <span className="material-symbols-outlined text-secondary text-[20px]">account_tree</span>
          <span className>EXPLORE ARCHITECTURE &amp; BENCHMARKS</span>
        </a>
      </div>
      {/* Central Interactive Verification AI Core & Modality Orbital Nodes */}
      <div className="relative w-full max-w-[960px] h-[480px] sm:h-[520px] flex items-center justify-center">
        {/* Rotating Holographic Calibration Rings (CSS-animated) */}
        <div className="absolute w-[340px] h-[340px] sm:w-[420px] sm:h-[420px] rounded-full animate-[spin_26s_linear_infinite] opacity-40 pointer-events-none" style={{background: 'radial-gradient(circle, transparent 62%, rgba(0, 216, 246, 0.25) 64%, transparent 68%), repeating-conic-gradient(from 0deg, rgba(0, 102, 255, 0.3) 0deg 15deg, transparent 15deg 30deg)'}} />
        <div className="absolute w-[280px] h-[280px] sm:w-[350px] sm:h-[350px] rounded-full animate-[spin_18s_linear_infinite_reverse] opacity-50 pointer-events-none" style={{background: 'repeating-conic-gradient(from 45deg, rgba(0, 216, 246, 0.4) 0deg 5deg, transparent 5deg 20deg)'}} />
        <div className="absolute w-[220px] h-[220px] sm:w-[280px] sm:h-[280px] rounded-full bg-cyan-border-glow/20 blur-2xl" />
        {/* Central Shield Core Container */}
        <div className="relative z-20 w-44 h-44 sm:w-56 sm:h-56 flex items-center justify-center p-3 rounded-full bg-surface-crisp shadow-[0_16px_40px_rgba(0,102,255,0.18)] transition-transform duration-500 hover:scale-105">
          <img alt="CheckReality AI Multimodal Shield" className="w-full h-full p-2 object-contain filter drop-shadow-[0_10px_20px_rgba(0,102,255,0.25)] select-none pointer-events-none" src="/logo.png" />
          <div className="absolute -bottom-3 px-space-sm py-0.5 rounded-full bg-verdict-real-bg shadow-sm flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-verdict-real animate-ping" />
            <span className="font-label-code text-label-caption text-verdict-real font-bold tracking-wider">BAYESIAN FUSED: 99.6%</span>
          </div>
        </div>
        {/* Orbital Nodes (Desktop 4 Stations around core) */}
        {/* Station 1: Text / RoBERTa */}
        <div className="absolute -top-2 left-4 sm:left-12 z-20 flex flex-col items-center group cursor-pointer" onClick={() => setActiveDemoTab('text')}>
          <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-surface-crisp shadow-md flex items-center justify-center transition-all duration-300 group-hover:shadow-[0_8px_24px_rgba(0,102,255,0.28)] group-hover:-translate-y-1">
            <span className="material-symbols-outlined text-primary text-[28px]">article</span>
          </div>
          <div className="mt-2 px-space-sm py-1 rounded-full bg-surface-crisp shadow-sm text-center">
            <div className="font-label-lg text-label-code text-on-surface font-bold">TEXT</div>
            <div className="font-label-caption text-[10px] text-secondary">RoBERTa-base</div>
          </div>
          {/* Pulse line toward center */}
          <div className="hidden md:block absolute top-1/2 left-full w-24 h-[2px] bg-gradient-to-r from-secondary-container to-transparent -rotate-12 pointer-events-none" />
        </div>
        {/* Station 2: Image / EfficientNet-B3 */}
        <div className="absolute -top-2 right-4 sm:right-12 z-20 flex flex-col items-center group cursor-pointer" onClick={() => setActiveDemoTab('image')}>
          <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-surface-crisp shadow-md flex items-center justify-center transition-all duration-300 group-hover:shadow-[0_8px_24px_rgba(0,102,255,0.28)] group-hover:-translate-y-1">
            <span className="material-symbols-outlined text-secondary text-[28px]">image</span>
          </div>
          <div className="mt-2 px-space-sm py-1 rounded-full bg-surface-crisp shadow-sm text-center">
            <div className="font-label-lg text-label-code text-on-surface font-bold">IMAGE</div>
            <div className="font-label-caption text-[10px] text-secondary">EfficientNet-B3</div>
          </div>
          <div className="hidden md:block absolute top-1/2 right-full w-24 h-[2px] bg-gradient-to-l from-secondary-container to-transparent rotate-12 pointer-events-none" />
        </div>
        {/* Station 3: Video / EfficientNet+LSTM */}
        <div className="absolute bottom-4 left-4 sm:left-14 z-20 flex flex-col items-center group cursor-pointer" onClick={() => setActiveDemoTab('video')}>
          <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-surface-crisp shadow-md flex items-center justify-center transition-all duration-300 group-hover:shadow-[0_8px_24px_rgba(0,102,255,0.28)] group-hover:-translate-y-1">
            <span className="material-symbols-outlined text-primary text-[28px]">video_file</span>
          </div>
          <div className="mt-2 px-space-sm py-1 rounded-full bg-surface-crisp shadow-sm text-center">
            <div className="font-label-lg text-label-code text-on-surface font-bold">VIDEO</div>
            <div className="font-label-caption text-[10px] text-secondary">EffNet + LSTM</div>
          </div>
          <div className="hidden md:block absolute bottom-1/2 left-full w-24 h-[2px] bg-gradient-to-r from-secondary-container to-transparent rotate-12 pointer-events-none" />
        </div>
        {/* Station 4: Audio / MelSpec CNN */}
        <div className="absolute bottom-4 right-4 sm:right-14 z-20 flex flex-col items-center group cursor-pointer" onClick={() => setActiveDemoTab('audio')}>
          <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-surface-crisp shadow-md flex items-center justify-center transition-all duration-300 group-hover:shadow-[0_8px_24px_rgba(0,102,255,0.28)] group-hover:-translate-y-1">
            <span className="material-symbols-outlined text-secondary text-[28px]">graphic_eq</span>
          </div>
          <div className="mt-2 px-space-sm py-1 rounded-full bg-surface-crisp shadow-sm text-center">
            <div className="font-label-lg text-label-code text-on-surface font-bold">AUDIO</div>
            <div className="font-label-caption text-[10px] text-secondary">MelSpec CNN</div>
          </div>
          <div className="hidden md:block absolute bottom-1/2 right-full w-24 h-[2px] bg-gradient-to-l from-secondary-container to-transparent -rotate-12 pointer-events-none" />
        </div>
      </div>
      {/* Quick Metrics Ribbon */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-space-md w-full max-w-4xl mt-space-lg">
        <div className="p-space-md rounded-2xl bg-surface-crisp shadow-sm flex flex-col items-center">
          <div className="font-label-metric text-label-metric text-primary">99.6%</div>
          <div className="font-label-caption text-label-caption text-on-surface-variant uppercase mt-1">Mean Benchmark Accuracy</div>
        </div>
        <div className="p-space-md rounded-2xl bg-surface-crisp shadow-sm flex flex-col items-center">
          <div className="font-label-metric text-label-metric text-primary">0.999+</div>
          <div className="font-label-caption text-label-caption text-on-surface-variant uppercase mt-1">Mean ROC-AUC Score</div>
        </div>
        <div className="p-space-md rounded-2xl bg-surface-crisp shadow-sm flex flex-col items-center">
          <div className="font-label-metric text-label-metric text-primary">4-in-1</div>
          <div className="font-label-caption text-label-caption text-on-surface-variant uppercase mt-1">Unified Modalities</div>
        </div>
        <div className="p-space-md rounded-2xl bg-surface-crisp shadow-sm flex flex-col items-center">
          <div className="font-label-metric text-label-metric text-primary">&lt; 0.6%</div>
          <div className="font-label-caption text-label-caption text-on-surface-variant uppercase mt-1">Mean False Positive Rate</div>
        </div>
      </div>
    </div>
  </section>
  {/* INTERACTIVE 6-STAGE PIPELINE ARCHITECTURE */}
  <section className="relative w-full py-space-3xl bg-surface-tint-card" id="pipeline-architecture">
    <div className="w-full max-w-[1280px] mx-auto px-margin flex flex-col">
      {/* Section Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between mb-space-2xl gap-space-md">
        <div className="flex flex-col">
          <div className="flex items-center gap-space-xs mb-space-xs">
            <span className="w-2.5 h-2.5 rounded-full bg-primary" />
            <span className="font-label-code text-label-caption text-primary font-bold uppercase tracking-wider">SYSTEM SPECIFICATION • TRD.MD</span>
          </div>
          <h2 className="font-headline-lg text-headline-lg text-on-surface tracking-tight">
            The 6-Stage Forensic Inference Pipeline
          </h2>
        </div>
        <div className="font-body-md text-body-md text-slate-body max-w-md">
          End-to-end synchronized deep neural architectures with mathematically proven Bayesian Multimodal Fusion and temperature-calibrated posterior probabilities.
        </div>
      </div>
      {/* 6-Stage Grid with Process Flow */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-space-lg">
        {/* Stage 1 */}
        <div className="relative p-space-lg rounded-2xl bg-surface-crisp shadow-sm hover:shadow-md transition-all flex flex-col justify-between group">
          <div className="flex items-center justify-between mb-space-md">
            <span className="font-label-code text-label-metric text-secondary-container font-black">01</span>
            <span className="px-space-sm py-0.5 rounded-full bg-surface-tint-ice font-label-code text-label-caption text-secondary">INGEST</span>
          </div>
          <div>
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-space-xs">FastAPI Asynchronous Ingestion</h3>
            <p className="font-body-sm text-body-sm text-slate-body mb-space-md">
              High-throughput asynchronous payload intake via FastAPI endpoints, supporting multi-part form uploads and immediate MongoDB registration.
            </p>
          </div>
          <div className="pt-space-sm flex items-center justify-between font-label-code text-label-caption text-on-surface-variant">
            <span className>Database: Motor (MongoDB)</span>
            <span className="text-verdict-real font-semibold">FastAPI Endpoints</span>
          </div>
        </div>
        {/* Stage 2 */}
        <div className="relative p-space-lg rounded-2xl bg-surface-crisp shadow-sm hover:shadow-md transition-all flex flex-col justify-between group">
          <div className="flex items-center justify-between mb-space-md">
            <span className="font-label-code text-label-metric text-secondary-container font-black">02</span>
            <span className="px-space-sm py-0.5 rounded-full bg-surface-tint-ice font-label-code text-label-caption text-secondary">PREPROCESS</span>
          </div>
          <div>
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-space-xs">Celery Background Workers</h3>
            <p className="font-body-sm text-body-sm text-slate-body mb-space-md">
              Decoupled background processing using Celery and Redis to handle heavy model inference tasks without blocking the main event loop.
            </p>
          </div>
          <div className="pt-space-sm flex items-center justify-between font-label-code text-label-caption text-on-surface-variant">
            <span className>Broker: Redis</span>
            <span className="text-verdict-real font-semibold">Celery Queue</span>
          </div>
        </div>
        {/* Stage 3 */}
        <div className="relative p-space-lg rounded-2xl bg-surface-crisp shadow-sm hover:shadow-md transition-all flex flex-col justify-between group">
          <div className="flex items-center justify-between mb-space-md">
            <span className="font-label-code text-label-metric text-secondary-container font-black">03</span>
            <span className="px-space-sm py-0.5 rounded-full bg-surface-tint-ice font-label-code text-label-caption text-secondary">INFERENCE</span>
          </div>
          <div>
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-space-xs">Python Model Adapter Layer</h3>
            <p className="font-body-sm text-body-sm text-slate-body mb-space-md">
              Extensible BaseModelAdapter architecture to seamlessly load and query trained PyTorch checkpoints (CrowdCore models) across all modalities.
            </p>
          </div>
          <div className="pt-space-sm flex items-center justify-between font-label-code text-label-caption text-on-surface-variant">
            <span className>Integration: PyTorch Checkpoints</span>
            <span className="text-primary font-semibold">Adapter Pattern</span>
          </div>
        </div>
        {/* Stage 4 */}
        <div className="relative p-space-lg rounded-2xl bg-surface-crisp shadow-sm hover:shadow-md transition-all flex flex-col justify-between group">
          <div className="flex items-center justify-between mb-space-md">
            <span className="font-label-code text-label-metric text-secondary-container font-black">04</span>
            <span className="px-space-sm py-0.5 rounded-full bg-surface-tint-ice font-label-code text-label-caption text-secondary">XAI</span>
          </div>
          <div>
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-space-xs">Explainable Evidence Extraction (XAI)</h3>
            <p className="font-body-sm text-body-sm text-slate-body mb-space-md">
              Generation of visual and textual explanations using SHAP for NLP token attribution and Grad-CAM for spatial image and video frame saliency.
            </p>
          </div>
          <div className="pt-space-sm flex items-center justify-between font-label-code text-label-caption text-on-surface-variant">
            <span className>Transparency: Model Attributions</span>
            <span className="text-secondary font-semibold">SHAP &amp; Grad-CAM</span>
          </div>
        </div>
        {/* Stage 5 */}
        <div className="relative p-space-lg rounded-2xl bg-surface-crisp shadow-sm hover:shadow-md transition-all flex flex-col justify-between group">
          <div className="flex items-center justify-between mb-space-md">
            <span className="font-label-code text-label-metric text-secondary-container font-black">05</span>
            <span className="px-space-sm py-0.5 rounded-full bg-surface-tint-ice font-label-code text-label-caption text-secondary">FUSION</span>
          </div>
          <div>
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-space-xs">Static Frontend Delivery</h3>
            <p className="font-body-sm text-body-sm text-slate-body mb-space-md">
              Production-ready React SPA built with Vite and Tailwind CSS, mounted as static assets directly within the FastAPI backend for unified deployment.
            </p>
          </div>
          <div className="pt-space-sm flex items-center justify-between font-label-code text-label-caption text-on-surface-variant">
            <span className>Port: 8001 (Unified)</span>
            <span className="text-verdict-real font-semibold">Vite + Tailwind</span>
          </div>
        </div>
        {/* Stage 6 */}
        <div className="relative p-space-lg rounded-2xl bg-surface-crisp shadow-sm hover:shadow-md transition-all flex flex-col justify-between group">
          <div className="flex items-center justify-between mb-space-md">
            <span className="font-label-code text-label-metric text-secondary-container font-black">06</span>
            <span className="px-space-sm py-0.5 rounded-full bg-surface-tint-ice font-label-code text-label-caption text-secondary">VERDICT</span>
          </div>
          <div>
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-space-xs">Real-Time UI Reflection</h3>
            <p className="font-body-sm text-body-sm text-slate-body mb-space-md">
              Live status polling fetching real-time JSON reports, displaying model execution results and inference scores on the React dashboard.
            </p>
          </div>
          <div className="pt-space-sm flex items-center justify-between font-label-code text-label-caption text-on-surface-variant">
            <span className>Format: REST JSON</span>
            <span className="text-primary font-semibold">React Hooks</span>
          </div>
        </div>
      </div>
    </div>
  </section>
  <section className="relative w-full py-space-xl bg-surface-crisp">
    <div className="w-full max-w-[1280px] mx-auto px-margin flex flex-col">
      {/* Academic & Institutional Attribution Card */}
      <div className="p-space-xl rounded-3xl bg-surface-tint-ice shadow-sm flex flex-col lg:flex-row items-center justify-between gap-space-xl mb-space-2xl">
        <div className="flex flex-col max-w-2xl">
          <div className="flex items-center gap-space-xs mb-space-xs">
            <span className="material-symbols-outlined text-primary text-[20px]">code</span>
            <span className="font-label-code text-label-caption text-primary font-bold uppercase tracking-wider">SYSTEM ARCHITECT &amp; LEAD RESEARCH</span>
          </div>
          <h3 className="font-headline-md text-headline-md text-on-surface mb-space-sm">Developed by <span className="text-primary font-bold">Divyansh Rajat</span></h3>
          <p className="font-body-md text-body-md text-slate-body">Engineered as an advanced deep multimodal forensic intelligence system. Implemented with mathematically calibrated Bayesian Multimodal Fusion, certainty-weighted ensemble modeling, and strict architectural adherence to the IEEE 830 System Requirements Specification.</p>
        </div>
        <div className="flex flex-col gap-space-sm shrink-0 w-full lg:w-auto">
          <div className="p-space-md rounded-2xl bg-surface-crisp shadow-sm flex items-center gap-space-md">
            <span className="material-symbols-outlined text-secondary text-[24px]">verified_user</span>
            <div>
              <div className="font-label-lg text-label-code text-on-surface font-bold">IEEE 830 COMPLIANT</div>
              <div className="font-label-caption text-label-caption text-slate-body">Forensic Software Requirements Spec</div>
            </div>
          </div>
          <div className="p-space-md rounded-2xl bg-surface-crisp shadow-sm flex items-center gap-space-md">
            <span className="material-symbols-outlined text-primary text-[24px]">terminal</span>
            <div>
              <div className="font-label-lg text-label-code text-on-surface font-bold">NEURAL PROVENANCE ENGINE</div>
              <div className="font-label-caption text-label-caption text-slate-body">Zero-Bias Evaluated Dataset Harness</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Non-Courtroom Decision Support Forensic Disclaimer */}
      <div className="p-space-md rounded-2xl bg-verdict-amber-bg flex items-start gap-space-md shadow-sm">
        <span className="material-symbols-outlined text-verdict-amber text-[24px] shrink-0 mt-0.5">gavel</span>
        <div className="flex flex-col">
          <span className="font-label-lg text-label-code text-on-surface font-bold">
            Decision-Support Forensics • Non-Courtroom Forensic Tool Notice
          </span>
          <p className="font-body-sm text-body-sm text-slate-body mt-1">
            CheckReality AI delivers probabilistic computational intelligence intended for newsroom verification, academic research, and content moderation triage. Diagnostic scores do not constitute definitive legal certification in statutory judicial proceedings without secondary human examiner verification.
          </p>
        </div>
      </div>
    </div>
  </section>
  {/* FINAL CTA STRIP */}
  <section className="relative w-full py-space-2xl bg-surface-container-high overflow-hidden">
    <div className="w-full max-w-[1280px] mx-auto px-margin flex flex-col md:flex-row items-center justify-between gap-space-lg relative z-10">
      <div className="flex flex-col max-w-xl text-center md:text-left">
        <h2 className="font-headline-lg text-headline-lg text-on-surface tracking-tight mb-space-xs">
          Ready to verify synthetic media in real time?
        </h2>
        <p className="font-body-md text-body-md text-slate-body">
          Deploy CheckReality AI directly in your investigative workflow or ingest multimodal inputs in our studio.
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-space-md shrink-0">
        <Link to="/upload" className="px-space-xl py-3 rounded-full bg-primary-container text-on-primary font-label-lg text-label-lg font-semibold hover:bg-electric-blue-hover shadow-[0_4px_16px_rgba(0,102,255,0.3)] transition-all">
          Launch Analysis Studio
        </Link>
        <a className="px-space-lg py-3 rounded-full bg-surface-crisp text-on-surface font-label-lg text-label-lg font-semibold hover:bg-surface-tint-ice transition-all shadow-sm" href="#pipeline-architecture">
          Review SRS Architecture
        </a>
      </div>
    </div>
  </section>
</div>

    </div>
  );
}
