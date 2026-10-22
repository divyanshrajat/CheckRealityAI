import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { submissionsService } from '../api/submissions';

interface Submission {
  _id: string;
  created_at: string;
  modalities: string[];
  status: string;
  result?: any;
}

export default function History() {
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const data = await submissionsService.getHistory();
        setSubmissions(data);
      } catch (e) {
        console.error("Failed to fetch history", e);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  return (
    <div className="w-full max-w-[1000px] mx-auto flex flex-col min-h-[60vh] p-8">
      <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight mb-8 text-center">Verification History</h1>
      
      {loading ? (
        <div className="flex justify-center text-primary">
          <span className="material-symbols-outlined animate-spin text-4xl">autorenew</span>
        </div>
      ) : submissions.length === 0 ? (
        <p className="font-body-lg text-body-lg text-slate-body text-center mt-8">
          You haven't uploaded any files for analysis yet.
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          {submissions.map((sub) => (
            <div key={sub._id} className="bg-surface-crisp p-4 rounded-xl shadow hover:bg-surface-tint-ice transition-all flex items-center justify-between">
              <Link to={`/report/${sub._id}`} className="flex flex-col flex-1 cursor-pointer">
                <span className="font-bold text-on-surface hover:text-primary transition-colors">Analysis ID: {sub._id.substring(0, 8)}...</span>
                <span className="text-slate-body text-sm">{new Date(sub.created_at).toLocaleString()}</span>
              </Link>
              <div className="flex items-center gap-4">
                <div className="flex gap-1.5 flex-wrap">
                  {(() => {
                    const rawMods = sub.modalities || [];
                    const setMods = new Set<string>();
                    
                    rawMods.forEach(m => {
                      if (m !== 'multimodal') setMods.add(m.toLowerCase());
                    });

                    const hasAudio = sub.result?.inference?.audio_stream !== undefined ||
                      sub.result?.inference?.audio !== undefined ||
                      sub.result?.fusion?.modality_contributions?.audio !== undefined;
                    
                    const hasVideo = sub.result?.inference?.video !== undefined ||
                      sub.result?.inference?.fake_probability !== undefined ||
                      sub.result?.fusion?.modality_contributions?.video !== undefined;

                    if (hasAudio) setMods.add('audio');
                    if (hasVideo) setMods.add('video');

                    if (setMods.size === 0) {
                      rawMods.forEach(m => setMods.add(m));
                    }

                    return Array.from(setMods).map(mod => (
                      <span 
                        key={mod} 
                        className={`px-2 py-0.5 rounded-full text-xs font-bold uppercase ${
                          mod === 'video' ? 'bg-blue-500/10 text-blue-600 border border-blue-500/20' :
                          mod === 'audio' ? 'bg-amber-500/10 text-amber-600 border border-amber-500/20' :
                          mod === 'image' ? 'bg-purple-500/10 text-purple-600 border border-purple-500/20' :
                          mod === 'text' ? 'bg-emerald-500/10 text-emerald-600 border border-emerald-500/20' :
                          'bg-primary/10 text-primary'
                        }`}
                      >
                        {mod}
                      </span>
                    ));
                  })()}
                </div>
                <div className={`px-3 py-1 rounded-full text-xs font-bold ${sub.status === 'completed' ? 'bg-verdict-real/20 text-verdict-real' : sub.status === 'failed' ? 'bg-red-500/20 text-red-500' : 'bg-secondary/20 text-secondary'}`}>
                  {sub.status.toUpperCase()}
                </div>
                
                <div className="flex items-center gap-1.5">
                  <Link to={`/report/${sub._id}`} className="p-2 text-primary hover:bg-primary/10 rounded-full transition-colors flex items-center justify-center" title="View Detailed Report">
                    <span className="material-symbols-outlined text-[20px]">visibility</span>
                  </Link>
                  {sub.status === 'completed' && (
                    <Link to={`/report/${sub._id}?view=certificate`} target="_blank" rel="noopener noreferrer" className="p-2 text-emerald-600 hover:bg-emerald-50 rounded-full transition-colors flex items-center justify-center" title="View Official Forensic Certificate">
                      <span className="material-symbols-outlined text-[20px]">verified</span>
                    </Link>
                  )}
                  <button 
                    onClick={async (e) => {
                      e.preventDefault();
                      if(window.confirm('Are you sure you want to delete this report?')) {
                        try {
                          await submissionsService.deleteSubmission(sub._id);
                          setSubmissions(submissions.filter(s => s._id !== sub._id));
                        } catch (err) {
                          console.error('Failed to delete', err);
                          alert('Failed to delete report.');
                        }
                      }
                    }} 
                    className="p-2 text-red-500 hover:bg-red-500/10 rounded-full transition-colors flex items-center justify-center" 
                    title="Delete Report"
                  >
                    <span className="material-symbols-outlined">delete</span>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
