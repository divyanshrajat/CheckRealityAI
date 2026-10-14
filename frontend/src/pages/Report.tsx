import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { submissionsService } from '../api/submissions';
import { ForensicMediaPlayer } from '../components/ForensicMediaPlayer';

export default function Report() {
  const { id } = useParams();
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(true);

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

  const fusion = report.result?.fusion;
  const xai = report.result?.xai;
  const isFake = fusion?.unified_fake_probability > 0.5;

  return (
    <div className="w-full flex flex-col">
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
            <button onClick={() => window.print()} className="ml-4 flex items-center gap-2 px-3 py-1 bg-primary text-on-primary rounded-full hover:bg-electric-blue-hover transition-colors font-bold text-sm" title="Download Report">
              <span className="material-symbols-outlined text-[16px]">download</span>
              Download PDF
            </button>
          </div>
        </section>

        <section className="w-full bg-surface-crisp rounded-xl shadow-md p-space-lg md:p-space-xl">
          <div className="grid grid-cols-1 gap-space-md">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className={`inline-flex items-center gap-space-xs px-space-md py-space-xs rounded-full font-label-lg text-label-lg shadow-sm ${isFake ? 'bg-verdict-fake-bg text-verdict-fake' : 'bg-verdict-real-bg text-verdict-real'}`}>
                <span className="material-symbols-outlined text-[20px]">{isFake ? 'gpp_bad' : 'gpp_good'}</span>
                <span>VERDICT: {fusion?.unified_label?.toUpperCase() || 'UNKNOWN'} (Confidence: {((fusion?.unified_fake_probability || 0) * 100).toFixed(1)}%)</span>
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
                <div className="flex items-center gap-2 font-label-code text-sm text-slate-body">
                  <span className="material-symbols-outlined text-primary">description</span>
                  <span className="font-bold text-on-surface">{report.original_filename}</span>
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
            </div>
          </section>
        )}

        <section className="flex flex-col gap-space-md">
          <h2 className="font-headline-sm text-headline-sm text-on-surface">Multi-Modal Breakdown</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-space-lg">
            {fusion?.modality_contributions && Object.entries(fusion.modality_contributions).map(([mod, prob]: [string, any]) => (
              <div key={mod} className="bg-surface-crisp rounded-xl p-space-lg shadow-sm flex flex-col justify-between">
                <div className="flex flex-col gap-space-sm">
                  <div className="flex items-center justify-between">
                    <span className="uppercase text-slate-body font-bold">{mod}</span>
                    <span className={prob > 0.5 ? 'text-verdict-fake font-bold' : 'text-verdict-real font-bold'}>{(prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-surface-container-high h-2 rounded-full overflow-hidden">
                    <div className={prob > 0.5 ? 'bg-verdict-fake h-full rounded-full' : 'bg-verdict-real h-full rounded-full'} style={{width: `${prob * 100}%`}} />
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
                 <p className="font-body-lg text-on-surface"><strong className="text-primary mr-2">Engine:</strong> {xai.xai_type || 'General Interpretability Engine'}</p>
                 <span className="px-3 py-1 rounded-full bg-surface-tint-ice text-primary text-xs font-bold uppercase tracking-wider">XAI DEEP-DIVE</span>
               </div>

               {/* Visual Anomalies */}
               {(xai.video?.suspicious_regions || xai.image?.suspicious_regions || xai.suspicious_regions) && (
                 <div className="mt-4 p-4 rounded-lg bg-verdict-fake-bg/30 border-l-4 border-verdict-fake">
                   <strong className="text-on-surface flex items-center gap-2 mb-2">
                     <span className="material-symbols-outlined text-verdict-fake">warning</span>
                     Visual Saliency Anomalies:
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
               
               {/* Audio Anomalies */}
               {(xai.audio?.suspicious_timestamps || xai.suspicious_timestamps) && (
                 <div className="mt-4 p-4 rounded-lg bg-verdict-amber-bg/30 border-l-4 border-verdict-amber">
                   <strong className="text-on-surface flex items-center gap-2 mb-2">
                     <span className="material-symbols-outlined text-verdict-amber">graphic_eq</span>
                     Acoustic Distortions Detected:
                   </strong>
                   <ul className="list-disc pl-6 flex flex-col gap-1 text-slate-body">
                     {(xai.audio?.suspicious_timestamps || xai.suspicious_timestamps).map((ts: any, i: number) => (
                       <li key={i}>{ts.reason} <span className="font-mono bg-surface-container px-1 py-0.5 rounded text-xs ml-1">({ts.start_sec}s - {ts.end_sec}s)</span></li>
                     ))}
                   </ul>
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
               
               {!xai.video?.suspicious_regions && !xai.image?.suspicious_regions && !xai.audio?.suspicious_timestamps && !xai.text?.suspicious_words && !xai.suspicious_regions && !xai.suspicious_timestamps && !xai.suspicious_words && (
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
  );
}
