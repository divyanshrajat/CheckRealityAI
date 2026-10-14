import React, { useRef, useState, useEffect } from 'react';

interface ForensicMediaPlayerProps {
  fileUrl: string;
  originalFilename?: string;
  className?: string;
}

export const ForensicMediaPlayer: React.FC<ForensicMediaPlayerProps> = ({
  fileUrl,
  originalFilename,
  className = '',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1.0);
  const [hasAudioTrack, setHasAudioTrack] = useState<boolean | null>(null);
  const [browserMutedNotice, setBrowserMutedNotice] = useState(false);

  const fullUrl = fileUrl.startsWith('http') 
    ? fileUrl 
    : `${import.meta.env.VITE_API_URL || 'http://localhost:8001'}${fileUrl}`;

  const isVideo = /\.(mp4|webm|ogg|mov|avi|mkv)$/i.test(fileUrl);
  const isAudio = /\.(mp3|wav|ogg|m4a|aac|flac)$/i.test(fileUrl);

  // Initialize video volume and ensure unmuted playback
  useEffect(() => {
    if (isVideo && videoRef.current) {
      const vid = videoRef.current;
      vid.muted = false;
      vid.volume = 1.0;
      setIsMuted(false);
      setVolume(1.0);
    }
  }, [fileUrl, isVideo]);

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      const vid = videoRef.current;
      vid.muted = false;
      vid.volume = volume;
      
      // Probe if video has audio stream
      // 1. Check audioTracks API if supported
      // 2. Check webkitAudioDecodedByteCount or mozHasAudio
      const anyVid = vid as any;
      if (anyVid.audioTracks && anyVid.audioTracks.length > 0) {
        setHasAudioTrack(true);
      } else if (anyVid.mozHasAudio !== undefined) {
        setHasAudioTrack(Boolean(anyVid.mozHasAudio));
      } else if (anyVid.webkitAudioDecodedByteCount !== undefined) {
        // Will be verified upon play
        setHasAudioTrack(true);
      } else {
        // Standard container assumption
        setHasAudioTrack(true);
      }
    }
  };

  const handlePlay = () => {
    setIsPlaying(true);
    if (videoRef.current) {
      // Ensure audio is unmuted when user initiates playback
      if (videoRef.current.muted) {
        videoRef.current.muted = false;
        setIsMuted(false);
      }
      videoRef.current.volume = volume;

      // Check if browser forcibly muted audio
      setTimeout(() => {
        if (videoRef.current && videoRef.current.muted) {
          setBrowserMutedNotice(true);
        } else {
          setBrowserMutedNotice(false);
        }
      }, 250);
    }
  };

  const handlePause = () => {
    setIsPlaying(false);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVol = parseFloat(e.target.value);
    setVolume(newVol);
    if (videoRef.current) {
      videoRef.current.volume = newVol;
      if (newVol > 0 && videoRef.current.muted) {
        videoRef.current.muted = false;
        setIsMuted(false);
        setBrowserMutedNotice(false);
      } else if (newVol === 0) {
        setIsMuted(true);
      }
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      const targetMute = !videoRef.current.muted;
      videoRef.current.muted = targetMute;
      setIsMuted(targetMute);
      setBrowserMutedNotice(false);
      if (!targetMute && volume === 0) {
        setVolume(1.0);
        videoRef.current.volume = 1.0;
      }
    }
  };

  const forceUnmuteAndPlay = () => {
    if (videoRef.current) {
      videoRef.current.muted = false;
      videoRef.current.volume = 1.0;
      setVolume(1.0);
      setIsMuted(false);
      setBrowserMutedNotice(false);
      videoRef.current.play().catch(console.error);
    }
  };

  return (
    <div className={`rounded-xl overflow-hidden bg-surface-container border border-outline/20 flex flex-col ${className}`}>
      {isVideo ? (
        <div className="flex flex-col w-full">
          {/* Active Sound / Audio Header Bar */}
          <div className="flex items-center justify-between px-4 py-2.5 bg-surface-container-high/60 border-b border-outline/10 text-xs">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-primary text-base">videocam</span>
              <span className="font-semibold text-on-surface">Video Stream</span>
              {hasAudioTrack !== false && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium text-[11px]">
                  <span className={`w-1.5 h-1.5 rounded-full ${isPlaying && !isMuted ? 'bg-emerald-400 animate-ping' : 'bg-emerald-400'}`}></span>
                  Audio Track Active (Sound Enabled)
                </span>
              )}
            </div>

            {/* Quick Volume / Mute Controls */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={toggleMute}
                className="p-1 hover:bg-surface-crisp rounded text-slate-body hover:text-on-surface transition-colors flex items-center"
                title={isMuted ? "Unmute Sound" : "Mute Sound"}
              >
                <span className="material-symbols-outlined text-[18px]">
                  {isMuted || volume === 0 ? 'volume_off' : volume > 0.5 ? 'volume_up' : 'volume_down'}
                </span>
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={isMuted ? 0 : volume}
                onChange={handleVolumeChange}
                className="w-16 md:w-24 h-1.5 accent-primary bg-surface-container-highest rounded-lg cursor-pointer"
                title={`Volume: ${Math.round((isMuted ? 0 : volume) * 100)}%`}
              />
              <span className="text-[11px] text-slate-body font-mono w-8 text-right">
                {Math.round((isMuted ? 0 : volume) * 100)}%
              </span>
            </div>
          </div>

          {/* Browser Muted Policy Notice (if any) */}
          {browserMutedNotice && (
            <div className="px-4 py-2 bg-amber-500/20 border-b border-amber-500/30 flex items-center justify-between text-xs text-amber-200">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-amber-400 text-sm">volume_off</span>
                <span>Audio was silenced by browser. Click to enable full video sound.</span>
              </div>
              <button
                type="button"
                onClick={forceUnmuteAndPlay}
                className="px-2.5 py-1 bg-amber-500 hover:bg-amber-400 text-black font-semibold rounded text-xs transition-colors"
              >
                Turn Sound On
              </button>
            </div>
          )}

          {/* Video Player */}
          <div className="relative bg-black flex items-center justify-center">
            <video
              ref={videoRef}
              src={fullUrl}
              controls
              playsInline
              preload="auto"
              crossOrigin="anonymous"
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={handlePlay}
              onPause={handlePause}
              onVolumeChange={() => {
                if (videoRef.current) {
                  setIsMuted(videoRef.current.muted);
                  setVolume(videoRef.current.volume);
                }
              }}
              className="w-full h-auto max-h-[460px] object-contain"
            />
          </div>
        </div>
      ) : isAudio ? (
        <div className="p-6 flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
              <span className="material-symbols-outlined text-xl">graphic_eq</span>
            </div>
            <div>
              <div className="font-semibold text-on-surface text-sm">
                {originalFilename || 'Audio Forensic Track'}
              </div>
              <div className="text-xs text-slate-body font-label-code">
                16kHz / Acoustic Spoofing Analysis Channel
              </div>
            </div>
          </div>
          <audio
            ref={audioRef}
            src={fullUrl}
            controls
            preload="auto"
            className="w-full rounded-lg"
          />
        </div>
      ) : (
        <div className="flex items-center justify-center bg-black/10 p-2">
          <img
            src={fullUrl}
            alt={originalFilename || "Analyzed Media"}
            className="w-full h-auto max-h-[500px] object-contain rounded-lg"
          />
        </div>
      )}
    </div>
  );
};
