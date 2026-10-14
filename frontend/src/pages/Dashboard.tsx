import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { submissionsService } from '../api/submissions';
import { ForensicMediaPlayer } from '../components/ForensicMediaPlayer';

export default function Dashboard() {
  const [latestSub, setLatestSub] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLatest = async () => {
      try {
        const data = await submissionsService.getHistory();
        if (data && data.length > 0) {
          setLatestSub(data[0]);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchLatest();
  }, []);

  if (loading) {
    return (
      <div className="w-full flex items-center justify-center min-h-[60vh]">
        <span className="material-symbols-outlined animate-spin text-4xl text-primary">sync</span>
      </div>
    );
  }

  if (!latestSub) {
    return (
      <div className="w-full flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <span className="material-symbols-outlined text-6xl text-outline">query_stats</span>
        <h2 className="font-headline-md text-headline-md text-on-surface">No Verifications Yet</h2>
        <p className="text-slate-body">Upload a file in the Studio to see the Fusion Engine in action.</p>
        <Link to="/upload" className="px-6 py-2 bg-primary text-on-primary rounded-full hover:bg-electric-blue-hover mt-4 font-bold">
          Go to Studio
        </Link>
      </div>
    );
  }

  const result = latestSub.result;
  const fusion = result?.fusion;
  const isFake = fusion?.unified_fake_probability > 0.5;

  return (
    <div className="w-full max-w-[1200px] mx-auto p-8 flex flex-col gap-8">
      <div className="flex flex-col gap-2">
        <h1 className="font-headline-lg text-headline-lg text-on-surface">Fusion Engine Results</h1>
        <p className="text-slate-body">Showing results for your most recent verification ({latestSub._id.substring(0, 8)}...)</p>
      </div>

      {latestSub.status !== 'completed' ? (
        <div className="bg-surface-crisp p-6 rounded-xl shadow-sm text-center">
          <span className="material-symbols-outlined animate-spin text-3xl text-primary mb-2">sync</span>
          <h3 className="font-headline-sm text-headline-sm text-on-surface">Processing...</h3>
          <p className="text-slate-body">Your file is currently being analyzed by the Fusion Engine.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Source Media Section */}
          <div className="bg-surface-crisp rounded-xl p-8 shadow-sm border border-outline/10 col-span-1 lg:col-span-2 flex flex-col gap-4">
             <h2 className="font-headline-sm text-headline-sm text-on-surface font-bold border-b border-outline/10 pb-4">Source Media Analyzed</h2>
             <div className="w-full">
                {latestSub.file_url ? (
                  <ForensicMediaPlayer
                    fileUrl={latestSub.file_url}
                    originalFilename={latestSub.original_filename}
                  />
                ) : (
                  <div className="text-slate-body p-8 text-center flex flex-col items-center gap-2 bg-black/5 rounded-lg">
                    <span className="material-symbols-outlined text-4xl">text_snippet</span>
                    <span>Text-only submission</span>
                  </div>
                )}
             </div>
          </div>

          {/* Fusion Verdict */}
          <div className="bg-surface-crisp rounded-xl p-8 shadow-sm border border-outline/10 flex flex-col gap-6">
            <h2 className="font-headline-sm text-headline-sm text-on-surface font-bold border-b border-outline/10 pb-4">Overall Unified Verdict</h2>
            
            <div className={`p-6 rounded-2xl flex flex-col items-center justify-center gap-4 text-center ${isFake ? 'bg-verdict-fake/10' : 'bg-verdict-real/10'}`}>
              <span className={`material-symbols-outlined text-6xl ${isFake ? 'text-verdict-fake' : 'text-verdict-real'}`}>
                {isFake ? 'warning' : 'verified_user'}
              </span>
              <div className="flex flex-col">
                <span className={`font-headline-lg text-headline-lg font-black ${isFake ? 'text-verdict-fake' : 'text-verdict-real'}`}>
                  {fusion?.unified_label?.toUpperCase() || 'UNKNOWN'}
                </span>
                <span className="text-on-surface-variant mt-2 font-bold">
                  Confidence: {((fusion?.unified_fake_probability || 0) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
            
            <div className="mt-4">
              <h3 className="font-label-lg font-bold text-on-surface mb-3">Modality Contributions:</h3>
              <div className="flex flex-col gap-3">
                {fusion?.modality_contributions && Object.entries(fusion.modality_contributions).map(([mod, prob]: [string, any]) => (
                  <div key={mod} className="flex flex-col gap-1">
                    <div className="flex justify-between font-label-code text-sm">
                      <span className="uppercase text-slate-body font-bold">{mod}</span>
                      <span className={prob > 0.5 ? 'text-verdict-fake font-bold' : 'text-verdict-real font-bold'}>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-surface-container rounded-full h-2 overflow-hidden">
                      <div className={`h-full rounded-full ${prob > 0.5 ? 'bg-verdict-fake' : 'bg-verdict-real'}`} style={{width: `${prob * 100}%`}}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Explainability (XAI) */}
          <div className="bg-surface-crisp rounded-xl p-8 shadow-sm border border-outline/10 flex flex-col gap-6">
             <h2 className="font-headline-sm text-headline-sm text-on-surface font-bold border-b border-outline/10 pb-4">Explainability (XAI) Details</h2>
             <div className="flex flex-col gap-4 text-slate-body">
                {result?.xai ? (
                  <>
                    {/* Visual Modalities (Image/Video) */}
                    {(result.xai.image?.suspicious_regions || result.xai.video?.suspicious_regions || result.xai.suspicious_regions) && (
                      <div className="mt-2 bg-surface-tint-ice p-4 rounded-xl border border-outline/10">
                        <strong className="text-on-surface flex items-center gap-2 mb-2"><span className="material-symbols-outlined text-primary">image_search</span> Visual Anomalies (Regions):</strong>
                        <ul className="list-disc pl-5 mt-2 flex flex-col gap-2">
                          {(result.xai.video?.suspicious_regions || result.xai.image?.suspicious_regions || result.xai.suspicious_regions).map((reg: any, i: number) => (
                            <li key={i}>{reg.reason} <span className="font-label-code text-xs text-primary bg-primary/10 px-2 rounded ml-2">x:{reg.x}, y:{reg.y}</span></li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Audio Modality */}
                    {(result.xai.audio?.suspicious_timestamps || result.xai.suspicious_timestamps) && (
                      <div className="mt-2 bg-surface-tint-ice p-4 rounded-xl border border-outline/10">
                        <strong className="text-on-surface flex items-center gap-2 mb-2"><span className="material-symbols-outlined text-secondary">graphic_eq</span> Audio Anomalies (Timestamps):</strong>
                        <ul className="list-disc pl-5 mt-2 flex flex-col gap-2">
                          {(result.xai.audio?.suspicious_timestamps || result.xai.suspicious_timestamps).map((ts: any, i: number) => (
                            <li key={i}>{ts.reason} <span className="font-label-code text-xs text-secondary bg-secondary/10 px-2 rounded ml-2">{ts.start_sec}s - {ts.end_sec}s</span></li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {/* Text Modality */}
                    {(result.xai.text?.suspicious_words || result.xai.suspicious_words) && (
                      <div className="mt-2 bg-surface-tint-ice p-4 rounded-xl border border-outline/10">
                        <strong className="text-on-surface flex items-center gap-2 mb-2"><span className="material-symbols-outlined text-verdict-amber">text_format</span> Flagged Vocabulary:</strong>
                        <div className="flex flex-wrap gap-2 mt-2">
                          {(result.xai.text?.suspicious_words || result.xai.suspicious_words).map((w: any, i: number) => (
                            <span key={i} className="px-2 py-1 bg-verdict-amber/20 text-verdict-amber rounded font-bold border border-verdict-amber/30">{w.word} ({w.importance_score})</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {!result.xai.image?.suspicious_regions && !result.xai.video?.suspicious_regions && !result.xai.audio?.suspicious_timestamps && !result.xai.text?.suspicious_words && !result.xai.suspicious_regions && !result.xai.suspicious_timestamps && !result.xai.suspicious_words && (
                      <div className="flex gap-3 bg-surface-tint-ice p-4 rounded-xl text-primary border border-primary/20">
                        <span className="material-symbols-outlined shrink-0 text-[24px]">info</span>
                        <p className="text-sm font-medium leading-relaxed">No explicit localization anomalies detected. The score relies on global ensemble confidence metrics.</p>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex gap-3 bg-surface-tint-ice p-4 rounded-xl text-slate-body border border-outline/10">
                    <span className="material-symbols-outlined shrink-0 text-[24px]">info</span>
                    <p className="text-sm">No explicit XAI data generated for this modality.</p>
                  </div>
                )}
             </div>
             
             <div className="mt-auto pt-6 border-t border-outline/10">
               <Link to={`/report/${latestSub._id}`} className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-surface-tint-ice text-primary font-bold hover:bg-primary hover:text-on-primary transition-all">
                 <span className="material-symbols-outlined">receipt_long</span>
                 View Full Detailed Report
               </Link>
             </div>
          </div>
        </div>
      )}
    </div>
  );
}
