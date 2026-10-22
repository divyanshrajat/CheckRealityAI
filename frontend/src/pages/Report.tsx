import { useState, useEffect } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { submissionsService } from '../api/submissions';
import { ForensicMediaPlayer } from '../components/ForensicMediaPlayer';
import { PrintableForensicReport } from '../components/PrintableForensicReport';
import { downloadCertificatePdf } from '../utils/pdfExport';

export default function Report() {
  const { id } = useParams();
  const [searchParams] = useSearchParams();
  const isCertificateOnly = searchParams.get('view') === 'certificate';
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownloadPdf = async () => {
    setIsDownloading(true);
    try {
      await downloadCertificatePdf('printable-forensic-report', `CheckRealityAI_Certificate_${id}.pdf`);
    } catch (e) {
      console.error('Download failed, using print fallback', e);
      window.print();
    } finally {
      setIsDownloading(false);
    }
  };

  useEffect(() => {
    if (!id) return;
    submissionsService.getReport(id).then(res => {
      setReport(res);
      setLoading(false);
    }).catch(e => {
      console.error(e);
      setLoading(false);
    });
  }, [id]);

  if (loading) {
    return <div className="p-10 text-center text-white">Loading forensic report...</div>;
  }
  if (!report) {
    return <div className="p-10 text-center text-red-500">Failed to load report.</div>;
  }

  if (isCertificateOnly) {
    return (
      <div className="min-h-screen bg-slate-200/90 py-6 px-4 flex justify-center items-start">
        <PrintableForensicReport report={report} id={id || ''} />
      </div>
    );
  }

  const fusion = report.result?.fusion;
  const xai = report.result?.xai;
  const inference = report.result?.inference;
  const isFake = fusion?.unified_fake_probability > 0.5;
  const rawProb = fusion?.unified_fake_probability ?? 0;
  const displayConfidence = fusion?.unified_confidence !== undefined
    ? Number(fusion.unified_confidence).toFixed(1)
    : (isFake ? (rawProb * 100) : ((1.0 - rawProb) * 100)).toFixed(1);

  const isVideoFile = !!(report.original_filename?.toLowerCase().match(/\.(mp4|mov|avi|webm|mkv|flv)$/) || report.modalities?.includes('video'));
  const isAudioFile = !!(report.original_filename?.toLowerCase().match(/\.(wav|mp3|m4a|aac|flac|ogg|opus)$/) || report.modalities?.includes('audio'));
  const isImageFile = !!(report.original_filename?.toLowerCase().match(/\.(jpg|jpeg|png|webp|bmp|tiff)$/) || report.modalities?.includes('image'));

  const modContributions = fusion?.modality_contributions || {};

  // Extract explicit stream probabilities
  const videoProb = modContributions.video !== undefined 
    ? modContributions.video 
    : (inference?.video?.fake_probability !== undefined 
        ? inference.video.fake_probability 
        : (inference?.fake_probability !== undefined && isVideoFile ? inference.fake_probability : null));

  const audioProb = modContributions.audio !== undefined 
    ? modContributions.audio 
    : (inference?.audio_stream?.spoof_probability !== undefined 
        ? inference.audio_stream.spoof_probability 
        : (inference?.audio?.spoof_probability !== undefined 
            ? inference.audio.spoof_probability 
            : (inference?.spoof_probability !== undefined && (isAudioFile || report.modalities?.includes('audio')) ? inference.spoof_probability : null)));

  const imageProb = modContributions.image !== undefined 
    ? modContributions.image 
    : (inference?.image?.fake_probability !== undefined 
        ? inference.image.fake_probability 
        : (inference?.fake_probability !== undefined && isImageFile ? inference.fake_probability : null));

  const textProb = modContributions.text !== undefined 
    ? modContributions.text 
    : (inference?.text?.misinformation_probability !== undefined 
        ? Math.max(inference.text.misinformation_probability, inference.text.ai_generated_probability || 0) 
        : (inference?.misinformation_probability !== undefined ? Math.max(inference.misinformation_probability, inference.ai_generated_probability || 0) : null));

  const hasAudioStream = audioProb !== null && audioProb !== undefined;
  const audioObj = inference?.audio_stream || inference?.audio || (isAudioFile ? inference : null);

  const streamCards: Array<{
    key: string;
    title: string;
    sublabel: string;
    model: string;
    prob: number;
    confidence: number;
    isFake: boolean;
    icon: string;
    tag?: string;
  }> = [];

  if (videoProb !== null && videoProb !== undefined) {
    const isFakeStream = videoProb > 0.5;
    const conf = isFakeStream ? (videoProb * 100) : ((1.0 - videoProb) * 100);
    streamCards.push({
      key: 'video',
      title: 'Video Facial Stream',
      sublabel: 'Temporal & Facial Spatial Dynamics',
      model: inference?.video?.model_version || inference?.model_version || 'EfficientNet-B0 + BiLSTM',
      prob: videoProb,
      confidence: conf,
      isFake: isFakeStream,
      icon: 'videocam',
      tag: 'PRIMARY VISUAL'
    });
  }

  if (audioProb !== null && audioProb !== undefined) {
    const isFakeStream = audioProb > 0.5;
    const conf = isFakeStream ? (audioProb * 100) : ((1.0 - audioProb) * 100);
    streamCards.push({
      key: 'audio',
      title: isVideoFile ? 'Demuxed Audio Track' : 'Acoustic Voice Stream',
      sublabel: isVideoFile ? 'Synchronized 16kHz PCM Stream' : 'Acoustic Log-Mel Spectrogram',
      model: audioObj?.model_version || 'ResNet-18 (SpecAugment)',
      prob: audioProb,
      confidence: conf,
      isFake: isFakeStream,
      icon: 'graphic_eq',
      tag: isVideoFile ? 'CONTAINER AUDIO' : 'PRIMARY AUDIO'
    });
  }

  if (imageProb !== null && imageProb !== undefined) {
    const isFakeStream = imageProb > 0.5;
    const conf = isFakeStream ? (imageProb * 100) : ((1.0 - imageProb) * 100);
    streamCards.push({
      key: 'image',
      title: 'Image Spatial Matrix',
      sublabel: 'Spatial Blending Artifacts',
      model: inference?.image?.model_version || 'EfficientNet-B3 (NoisyStudent)',
      prob: imageProb,
      confidence: conf,
      isFake: isFakeStream,
      icon: 'image',
      tag: 'SPATIAL IMAGE'
    });
  }

  if (textProb !== null && textProb !== undefined) {
    const isFakeStream = textProb > 0.5;
    const conf = isFakeStream ? (textProb * 100) : ((1.0 - textProb) * 100);
    streamCards.push({
      key: 'text',
      title: 'Text Semantic Stream',
      sublabel: 'Semantic Framing & Synthetic Style',
      model: inference?.text?.model_version || 'Dual RoBERTa-base',
      prob: textProb,
      confidence: conf,
      isFake: isFakeStream,
      icon: 'article',
      tag: 'NLP SEMANTICS'
    });
  }

  if (streamCards.length === 0 && Object.keys(modContributions).length > 0) {
    Object.entries(modContributions).forEach(([mod, prob]: [string, any]) => {
      const isFakeStream = prob > 0.5;
      const conf = isFakeStream ? (prob * 100) : ((1.0 - prob) * 100);
      streamCards.push({
        key: mod,
        title: `${mod.toUpperCase()} Stream`,
        sublabel: 'Forensic Signal Evaluation',
        model: 'Calibrated Detector',
        prob: prob,
        confidence: conf,
        isFake: isFakeStream,
        icon: 'analytics'
      });
    });
  }

  return (
    <>
      {/* 1. Official Single-Page Printable Forensic Report (Active during Print / PDF Export) */}
      <div className="print-only">
        <PrintableForensicReport report={report} id={id || ''} />
      </div>

      {/* 2. Interactive Screen View */}
      <div className="interactive-screen-report w-full flex flex-col">
        <div className="w-full max-w-[1280px] mx-auto px-margin-mobile md:px-margin py-space-xl flex flex-col gap-space-2xl relative">
          <section className="flex flex-col md:flex-row md:items-center justify-between gap-space-md pb-space-xs">
            <div className="flex flex-col gap-space-xs">
              <div className="flex items-center gap-space-xs">
                <span className="px-space-sm py-space-xs rounded-full bg-surface-tint-ice text-primary font-label-code text-label-code font-bold tracking-wide">
                  REPORT ID: #{id}
                </span>
              </div>
              <div className="flex flex-wrap items-baseline gap-x-space-md gap-y-space-xs">
                <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight">Forensic Authenticity Report</h1>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-space-sm bg-surface-crisp px-space-md py-space-sm rounded-xl shadow-sm">
              <div className="flex items-center gap-space-xs text-tertiary">
                <span className="material-symbols-outlined text-[18px]">schedule</span>
                <span className="font-body-sm text-body-sm">{new Date(report.created_at || Date.now()).toLocaleString()}</span>
              </div>
              <button 
                type="button"
                onClick={() => setShowPreviewModal(true)} 
                className="ml-2 flex items-center gap-1.5 px-3 py-1 bg-surface-tint-ice text-primary border border-primary/20 rounded-full hover:bg-surface-tint-ice/80 transition-colors font-bold text-sm" 
                title="Preview Standard Single-Page Certificate"
              >
                <span className="material-symbols-outlined text-[16px]">visibility</span>
                Preview PDF
              </button>
              <button 
                type="button"
                onClick={handleDownloadPdf}
                disabled={isDownloading}
                className="flex items-center gap-2 px-3.5 py-1 bg-primary text-on-primary rounded-full hover:bg-electric-blue-hover transition-colors font-bold text-sm shadow-[0_2px_8px_rgba(0,102,255,0.25)] disabled:opacity-60 cursor-pointer" 
                title="Download Single-Page PDF Report Directly"
              >
                <span className={`material-symbols-outlined text-[16px] ${isDownloading ? 'animate-spin' : ''}`}>
                  {isDownloading ? 'sync' : 'download'}
                </span>
                {isDownloading ? 'Generating PDF...' : 'Download PDF'}
              </button>
            </div>
          </section>

        <section className="w-full bg-surface-crisp rounded-xl shadow-md p-space-lg md:p-space-xl">
          <div className="grid grid-cols-1 gap-space-md">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className={`inline-flex items-center gap-space-xs px-space-md py-space-xs rounded-full font-label-lg text-label-lg shadow-sm ${isFake ? 'bg-verdict-fake-bg text-verdict-fake' : 'bg-verdict-real-bg text-verdict-real'}`}>
                <span className="material-symbols-outlined text-[20px]">{isFake ? 'gpp_bad' : 'gpp_good'}</span>
                <span>VERDICT: {fusion?.unified_label?.toUpperCase() || 'UNKNOWN'} (Confidence: {displayConfidence}%)</span>
              </div>
              {fusion?.risk_severity && (
                <div className="flex items-center gap-2">
                  <span className="text-xs uppercase font-bold text-slate-body">Risk Tier:</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                    fusion.risk_severity.includes('CRITICAL') ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                    fusion.risk_severity.includes('HIGH') ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                    fusion.risk_severity.includes('MODERATE') ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                    'bg-green-500/20 text-green-400 border border-green-500/30'
                  }`}>
                    {fusion.risk_severity}
                  </span>
                  {fusion.cross_modal_agreement !== undefined && (
                    <span className="text-xs text-tertiary ml-2">
                      Agreement: {(fusion.cross_modal_agreement * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              )}
            </div>
            {fusion?.forensic_summary && (
              <div className="mt-2 p-4 bg-surface-container-low rounded-lg border border-outline/10 text-body-md text-on-surface leading-relaxed">
                <div className="flex items-center gap-2 text-primary font-bold text-sm mb-1">
                  <span className="material-symbols-outlined text-[18px]">psychology</span>
                  Forensic Synthesis Engine
                </div>
                <p className="text-slate-body text-sm">{fusion.forensic_summary}</p>
              </div>
            )}
          </div>
        </section>


        {/* Source Media Display */}
        {(report.file_url || report.text_content) && (
          <section className="flex flex-col gap-space-md">
            <h2 className="font-headline-sm text-headline-sm text-on-surface">Source Media Analyzed</h2>
            <div className="bg-surface-crisp rounded-xl p-space-lg shadow-sm border border-outline/10 flex flex-col gap-4">
              {report.original_filename && (
                <div className="flex items-center justify-between flex-wrap gap-2 font-label-code text-sm text-slate-body">
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-primary">description</span>
                    <span className="font-bold text-on-surface">{report.original_filename}</span>
                  </div>
                  {isVideoFile && (
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold font-mono flex items-center gap-1 ${
                      hasAudioStream ? 'bg-primary/10 text-primary border border-primary/20' : 'bg-surface-container text-tertiary'
                    }`}>
                      <span className="material-symbols-outlined text-[14px]">
                        {hasAudioStream ? 'volume_up' : 'volume_off'}
                      </span>
                      {hasAudioStream ? 'VIDEO + DEMUXED AUDIO (DUAL STREAM)' : 'VIDEO ONLY (NO AUDIO TRACK)'}
                    </span>
                  )}
                </div>
              )}
              {report.file_url && (
                <ForensicMediaPlayer
                  fileUrl={report.file_url}
                  originalFilename={report.original_filename}
                />
              )}
              {report.text_content && (
                <div className="p-4 bg-surface-container-low rounded-lg text-body-md text-on-surface border border-outline/10 whitespace-pre-wrap">
                  {report.text_content}
                </div>
              )}

              {/* Synchronized Audio Detection Box for Video Files */}
              {isVideoFile && (
                <div className={`p-3 rounded-lg border flex items-center justify-between text-xs transition-colors ${
                  hasAudioStream ? 'bg-primary/5 border-primary/20' : 'bg-surface-container-low border-outline/10'
                }`}>
                  <div className="flex items-center gap-2.5">
                    <span className="material-symbols-outlined text-primary text-[20px]">
                      {hasAudioStream ? 'graphic_eq' : 'volume_off'}
                    </span>
                    <div>
                      <div className="font-bold text-on-surface">
                        {hasAudioStream ? 'Audio Track Demuxed & Separately Analyzed' : 'Audio Track Detection'}
                      </div>
                      <div className="text-slate-body text-[11px]">
                        {hasAudioStream 
                          ? 'Extracted 16kHz mono PCM waveform evaluated with ResNet-18 SpecAugment + Mel Spectrogram Gradient Profiler' 
                          : 'No synchronized acoustic channel found in video container; visual temporal pipeline applied.'}
                      </div>
                    </div>
                  </div>
                  {hasAudioStream && audioProb !== null && (
                    <span className={`px-2.5 py-1 rounded-full font-mono font-bold text-xs ${
                      audioProb > 0.5 
                        ? 'bg-verdict-fake/10 text-verdict-fake border border-verdict-fake/20' 
                        : 'bg-verdict-real/10 text-verdict-real border border-verdict-real/20'
                    }`}>
                      {audioProb > 0.5 ? 'AUDIO: MANIPULATION FLAGGED' : 'AUDIO: BONAFIDE SPEECH'}
                    </span>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

        {/* Multi-Modal Breakdown (Dedicated Cards for Every Active Stream) */}
        <section className="flex flex-col gap-space-md">
          <div className="flex items-center justify-between">
            <h2 className="font-headline-sm text-headline-sm text-on-surface">Multi-Modal Breakdown</h2>
            <span className="text-xs text-tertiary font-mono">
              {streamCards.length} {streamCards.length === 1 ? 'STREAM' : 'STREAMS'} EVALUATED
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-space-lg">
            {streamCards.map((sc) => (
              <div key={sc.key} className="bg-surface-crisp rounded-xl p-space-lg shadow-sm border border-outline/10 flex flex-col justify-between hover:shadow-md transition-shadow">
                <div className="flex flex-col gap-space-sm">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      <span className="material-symbols-outlined text-primary text-[20px]">{sc.icon}</span>
                      <div className="flex flex-col">
                        <span className="text-xs font-bold text-slate-body uppercase">{sc.title}</span>
                        <span className="text-[10px] text-tertiary">{sc.sublabel}</span>
                      </div>
                    </div>
                    {sc.tag && (
                      <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-surface-container text-tertiary font-bold">
                        {sc.tag}
                      </span>
                    )}
                  </div>

                  <div className="mt-2 flex items-baseline justify-between">
                    <span className={`text-base font-black ${sc.isFake ? 'text-verdict-fake' : 'text-verdict-real'}`}>
                      {sc.confidence.toFixed(1)}% {sc.isFake ? 'Synthetic' : 'Authentic'}
                    </span>
                    <span className="text-xs font-mono text-tertiary">
                      P: {sc.prob.toFixed(4)}
                    </span>
                  </div>

                  <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                    <div 
                      className={sc.isFake ? 'bg-verdict-fake h-full rounded-full transition-all duration-500' : 'bg-verdict-real h-full rounded-full transition-all duration-500'} 
                      style={{ width: `${sc.confidence}%` }} 
                    />
                  </div>

                  <div className="mt-1 pt-2 border-t border-outline/10 flex items-center justify-between text-[11px] text-slate-body">
                    <span className="font-semibold text-tertiary">Model:</span>
                    <span className="font-mono text-on-surface font-semibold truncate ml-1" title={sc.model}>
                      {sc.model}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {xai && (
          <section className="flex flex-col gap-space-md">
            <h2 className="font-headline-sm text-headline-sm text-on-surface">Explainability (XAI) Flags</h2>
            <div className="bg-surface-crisp rounded-xl p-space-lg shadow-sm border border-outline/10 hover:shadow-md transition-all">
               
               <div className="flex items-center justify-between mb-4">
                 <p className="font-body-lg text-on-surface">
                   <strong className="text-primary mr-2">Engine:</strong>
                   {xai.xai_type || (hasAudioStream && isVideoFile ? 'Temporal Grad-CAM + Acoustic Mel Profiler' : xai.audio?.xai_type || xai.video?.xai_type || xai.image?.xai_type || xai.text?.xai_type || 'General Interpretability Engine')}
                 </p>
                 <span className="px-3 py-1 rounded-full bg-surface-tint-ice text-primary text-xs font-bold uppercase tracking-wider">XAI DEEP-DIVE</span>
               </div>

               {/* Visual Anomalies */}
               {(xai.video?.suspicious_regions || xai.image?.suspicious_regions || xai.suspicious_regions) && (
                 <div className="mt-4 p-4 rounded-lg bg-verdict-fake-bg/30 border-l-4 border-verdict-fake">
                   <strong className="text-on-surface flex items-center gap-2 mb-2">
                     <span className="material-symbols-outlined text-verdict-fake">warning</span>
                     Visual Saliency Anomalies (Grad-CAM):
                   </strong>
                   <ul className="list-disc pl-6 flex flex-col gap-1 text-slate-body">
                     {(xai.video?.suspicious_regions || xai.image?.suspicious_regions || xai.suspicious_regions).map((reg: any, i: number) => (
                       <li key={i}>{reg.reason} <span className="text-tertiary text-xs">(x:{reg.x}, y:{reg.y})</span></li>
                     ))}
                   </ul>
                   {(xai.video?.heatmap_base64 || xai.image?.heatmap_base64 || xai.heatmap_base64) && (
                     <div className="mt-4 max-w-[400px] rounded-lg overflow-hidden border border-outline shadow-sm relative group">
                       <img src={`data:image/jpeg;base64,${(xai.video?.heatmap_base64 || xai.image?.heatmap_base64 || xai.heatmap_base64)}`} alt="Grad-CAM Heatmap" className="w-full h-auto object-cover" />
                       <div className="absolute top-2 right-2 px-2 py-1 rounded text-xs font-bold bg-verdict-fake text-white">HEATMAP</div>
                     </div>
                   )}
                 </div>
               )}
               
               {/* Audio Anomalies / Saliency (Standalone audio or demuxed from video) */}
               {(xai.audio?.suspicious_timestamps || xai.suspicious_timestamps || hasAudioStream || (xai.audio && !xai.audio.error)) && (
                 <div className={`mt-4 p-4 rounded-lg border-l-4 ${
                   (audioProb !== null ? audioProb > 0.5 : isFake) 
                     ? 'bg-verdict-amber-bg/30 border-verdict-amber' 
                     : 'bg-verdict-real-bg/30 border-verdict-real'
                 }`}>
                   <div className="flex items-center justify-between mb-2">
                     <strong className="text-on-surface flex items-center gap-2">
                       <span className={`material-symbols-outlined ${(audioProb !== null ? audioProb > 0.5 : isFake) ? 'text-verdict-amber' : 'text-verdict-real'}`}>
                         {(audioProb !== null ? audioProb > 0.5 : isFake) ? 'graphic_eq' : 'verified'}
                       </span>
                       {isVideoFile ? 'Demuxed Audio Acoustic Saliency (ResNet-18):' : 'Acoustic Saliency & Formant Consistency (ResNet-18):'}
                     </strong>
                     {audioProb !== null && (
                       <span className={`text-xs px-2 py-0.5 rounded font-mono font-bold ${
                         audioProb > 0.5 ? 'bg-verdict-amber/20 text-verdict-amber' : 'bg-verdict-real/20 text-verdict-real'
                       }`}>
                         {audioProb > 0.5 ? 'VOICE SPOOF DETECTED' : 'BONAFIDE SPEECH CONFIRMED'}
                       </span>
                     )}
                   </div>

                   {(xai.audio?.suspicious_timestamps || xai.suspicious_timestamps)?.length > 0 ? (
                     <ul className="list-disc pl-6 flex flex-col gap-1 text-slate-body">
                       {(xai.audio?.suspicious_timestamps || xai.suspicious_timestamps).map((ts: any, i: number) => (
                         <li key={i}>
                           {ts.reason} <span className="font-mono bg-surface-container px-1 py-0.5 rounded text-xs ml-1">({ts.start_sec}s - {ts.end_sec}s)</span>
                         </li>
                       ))}
                     </ul>
                   ) : (
                     <p className="text-slate-body text-sm">
                       {xai.audio?.note || (
                         audioProb !== null && audioProb > 0.5
                           ? 'Mel-spectrogram gradient energy highlights synthetic vocoder phase discontinuities and high-frequency acoustic distortion.'
                           : 'Continuous harmonic formant progression verified across all time-frequency bins. Biological vocal tract resonance confirmed with zero vocoder phase splices.'
                       )}
                     </p>
                   )}
                 </div>
               )}

               {/* Text Anomalies */}
               {(xai.text?.suspicious_words || xai.suspicious_words) && (
                 <div className="mt-4 p-4 rounded-lg bg-verdict-real-bg/30 border-l-4 border-verdict-real">
                   <strong className="text-on-surface flex items-center gap-2 mb-2">
                     <span className="material-symbols-outlined text-verdict-real">article</span>
                     Semantic Flags (Tokens):
                   </strong>
                   <div className="flex flex-wrap gap-2 mt-2">
                     {(xai.text?.suspicious_words || xai.suspicious_words).map((w: any, i: number) => (
                       <span key={i} className="px-2 py-1 bg-surface-container text-on-surface rounded font-medium border border-outline/20">
                         {w.word}
                       </span>
                     ))}
                   </div>
                 </div>
               )}
               
               {!xai.video?.suspicious_regions && !xai.image?.suspicious_regions && !xai.audio?.suspicious_timestamps && !xai.text?.suspicious_words && !xai.suspicious_regions && !xai.suspicious_timestamps && !xai.suspicious_words && !hasAudioStream && (
                 <div className="mt-4 p-4 rounded-lg bg-surface-tint-ice/50 border border-primary/20">
                   <p className="text-slate-body text-sm flex items-center gap-2">
                     <span className="material-symbols-outlined text-primary">info</span>
                     No explicit localization anomalies detected. The score relies on global ensemble confidence metrics.
                   </p>
                 </div>
               )}
            </div>
          </section>
        )}
      </div>
    </div>

      {/* 3. Certificate Preview Modal */}
      {showPreviewModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 overflow-y-auto no-print">
          <div className="bg-surface-crisp border border-outline/20 rounded-2xl max-w-4xl w-full max-h-[92vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="px-6 py-4 bg-surface-container-high flex items-center justify-between border-b border-outline/10">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary text-xl">description</span>
                <h3 className="font-headline-sm text-base text-on-surface font-bold">Official Forensic Certificate Preview (A4 Single Page)</h3>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleDownloadPdf}
                  disabled={isDownloading}
                  className="flex items-center gap-1.5 px-4 py-1.5 bg-primary text-on-primary font-bold text-xs rounded-full hover:bg-electric-blue-hover transition-colors shadow-sm disabled:opacity-60 cursor-pointer"
                  title="Download Certificate PDF Directly"
                >
                  <span className={`material-symbols-outlined text-sm ${isDownloading ? 'animate-spin' : ''}`}>
                    {isDownloading ? 'sync' : 'download'}
                  </span>
                  {isDownloading ? 'Downloading...' : 'Download PDF'}
                </button>
                <button
                  type="button"
                  onClick={() => window.print()}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 bg-surface-container-low text-on-surface font-bold text-xs rounded-full hover:bg-surface-container transition-colors border border-outline/20 cursor-pointer"
                  title="Open Print Dialog"
                >
                  <span className="material-symbols-outlined text-sm">print</span>
                  Print
                </button>
                <button
                  type="button"
                  onClick={() => setShowPreviewModal(false)}
                  className="p-1.5 hover:bg-surface-container rounded-lg text-slate-body hover:text-on-surface transition-colors"
                  title="Close Preview"
                >
                  <span className="material-symbols-outlined text-xl">close</span>
                </button>
              </div>
            </div>

            <div className="p-6 overflow-y-auto bg-slate-900/60 flex items-start justify-center">
              <div className="bg-white rounded-lg shadow-xl overflow-hidden w-full max-w-[800px]">
                <PrintableForensicReport report={report} id={id || ''} />
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
