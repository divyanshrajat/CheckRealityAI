// @ts-nocheck
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { authService } from '../api/auth';

export default function Signup() {
  const [fullName, setFullName] = useState('');
  const [organization, setOrganization] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const getPasswordStrength = (pass: string) => {
    let score = 0;
    if (pass.length >= 8) score += 1;
    if (/[0-9]/.test(pass)) score += 1;
    if (/[^A-Za-z0-9]/.test(pass)) score += 1;
    if (/[A-Z]/.test(pass)) score += 1; 
    return score;
  };

  const strengthScore = getPasswordStrength(password);
  const hasLength = password.length >= 8;
  const hasNum = /[0-9]/.test(password);
  const hasSym = /[^A-Za-z0-9]/.test(password);

  let strengthLabel = 'Awaiting Input';
  let strengthColor = 'text-slate-body';
  let barColors = ['bg-surface-container-highest', 'bg-surface-container-highest', 'bg-surface-container-highest', 'bg-surface-container-highest'];

  if (password.length > 0) {
    if (strengthScore <= 1) {
      strengthLabel = 'Weak';
      strengthColor = 'text-red-500';
      barColors = ['bg-red-500', 'bg-surface-container-highest', 'bg-surface-container-highest', 'bg-surface-container-highest'];
    } else if (strengthScore === 2) {
      strengthLabel = 'Fair';
      strengthColor = 'text-amber-500';
      barColors = ['bg-amber-500', 'bg-amber-500', 'bg-surface-container-highest', 'bg-surface-container-highest'];
    } else if (strengthScore === 3) {
      strengthLabel = 'Good';
      strengthColor = 'text-blue-500';
      barColors = ['bg-blue-500', 'bg-blue-500', 'bg-blue-500', 'bg-surface-container-highest'];
    } else {
      strengthLabel = 'Strong';
      strengthColor = 'text-emerald-500';
      barColors = ['bg-emerald-500', 'bg-emerald-500', 'bg-emerald-500', 'bg-emerald-500'];
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    setError('');
    setIsLoading(true);
    try {
      const res = await authService.signup(email, password, fullName, organization);
      login(res.access_token, res.user);
      navigate('/');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create account');
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col w-full relative min-h-[calc(100vh-4rem)] justify-center items-center py-space-xl px-margin-mobile md:px-margin overflow-hidden">
  {/* Luminous Electric Digital Wave & Neural Circuit Ambient Canvas */}
  <div className="absolute inset-0 pointer-events-none -z-10 flex items-center justify-center overflow-hidden">
    {/* Fluid Wave Filaments (SVG Pure Light Vector) */}
    <svg className="absolute w-[180%] max-w-none h-full opacity-60 mix-blend-multiply" fill="none" viewBox="0 0 1600 900" xmlns="http://www.w3.org/2000/svg">
      <path d="M-100 700 C 300 850, 600 550, 950 680 C 1300 810, 1500 620, 1800 750" stroke="url(#electricWaveGrad1)" strokeDasharray="6 4" strokeOpacity="0.8" strokeWidth="2.5" />
      <path d="M-120 740 C 280 890, 580 590, 930 720 C 1280 850, 1480 660, 1780 790" stroke="url(#electricWaveGrad2)" strokeOpacity="0.5" strokeWidth="1.5" />
      <path d="M-50 400 C 400 250, 750 600, 1100 380 C 1450 160, 1600 450, 1750 320" stroke="url(#electricWaveGrad3)" strokeOpacity="0.45" strokeWidth="1.8" />
      {/* Soft Cyan Fluid Glow Blurs */}
      <path d="M-200 650 Q 400 920 800 650 T 1800 700 L 1800 900 L -200 900 Z" fill="url(#cyanUnderflow)" opacity="0.25" />
      <defs>
        <linearGradient id="electricWaveGrad1" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#00D8F6" stopOpacity="0.1" />
          <stop offset="40%" stopColor="#0066FF" stopOpacity="0.7" />
          <stop offset="70%" stopColor="#00D8F6" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#0050CB" stopOpacity="0.1" />
        </linearGradient>
        <linearGradient id="electricWaveGrad2" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#0050CB" stopOpacity="0.05" />
          <stop offset="50%" stopColor="#00D8F6" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#0066FF" stopOpacity="0.05" />
        </linearGradient>
        <linearGradient id="electricWaveGrad3" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#00D8F6" stopOpacity="0.1" />
          <stop offset="55%" stopColor="#006688" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#00D8F6" stopOpacity="0.1" />
        </linearGradient>
        <linearGradient id="cyanUnderflow" x1="50%" x2="50%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#00D8F6" />
          <stop offset="100%" stopColor="#0050CB" stopOpacity={0} />
        </linearGradient>
      </defs>
    </svg>
    {/* Floating Cyan Verification Glow Spheres */}
    <div className="absolute -top-32 left-1/4 w-96 h-96 rounded-full bg-cyan-energy/10 blur-3xl" />
    <div className="absolute bottom-10 right-1/4 w-[28rem] h-[28rem] rounded-full bg-primary/10 blur-3xl" />
  </div>
  {/* Registration Interactive Card Shell */}
  <div className="w-full max-w-[620px] bg-surface-crisp/95 backdrop-blur-2xl rounded-2xl shadow-[0_20px_50px_-15px_rgba(0,80,203,0.12),0_0_0_1px_rgba(186,230,253,0.8),0_0_28px_0_rgba(0,216,246,0.15)] p-space-md sm:p-space-xl flex flex-col relative">
    <div className="flex items-center justify-between pb-space-sm mb-space-sm">
      <div className="inline-flex items-center gap-space-xs px-2.5 py-1 rounded-full bg-surface-tint-ice text-primary font-label-caption text-label-caption">
        <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
        <span>MULTI-MODAL VERIFICATION PORTAL</span>
      </div>
    </div>
    {/* Header Section with Official Platform Logo */}
    <div className="flex flex-col items-center text-center mb-space-lg">
      <div className="relative mb-space-xs group">
        <div className="absolute -inset-2 bg-gradient-to-r from-cyan-energy to-primary rounded-2xl blur-lg opacity-30 group-hover:opacity-60 transition duration-500" />
        <div className="relative w-16 h-16 rounded-xl bg-surface-crisp shadow-md flex items-center justify-center p-2">
          <img alt="CheckReality AI Logo" className="w-full h-full object-contain" src="/logo.png" />
        </div>
      </div>
      <h1 className="font-headline-md text-headline-md text-on-surface tracking-tight mt-space-xs">
        Create Analyst Account
      </h1>
      <p className="font-body-md text-body-md text-slate-body mt-1 max-w-[480px]">
        Access real-time multimodal deepfake detection and cryptographic truth verification.
      </p>
    </div>
    {/* Interactive Registration Form */}
    {error && (
      <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 font-label-md text-label-md text-center">
        {error}
      </div>
    )}
    <form className="flex flex-col gap-space-md" id="signup-form" onSubmit={handleSubmit}>
      {/* Field 1 & 3 Dual Column Grid (Name & Organization) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
        {/* Field 1: Full Name */}
        <div className="flex flex-col gap-1.5">
          <label className="font-label-lg text-label-lg text-on-surface flex items-center justify-between" htmlFor="full-name">
            <span>Full Name</span>
            <span className="font-label-code text-label-code text-secondary text-[11px]">REQUIRED</span>
          </label>
          <div className="relative flex items-center">
            <span className="material-symbols-outlined absolute left-3 text-slate-body text-[18px]">badge</span>
            <input value={fullName} onChange={(e) => setFullName(e.target.value)} className="w-full pl-10 pr-3.5 py-2.5 rounded-lg bg-surface-crisp text-on-surface font-body-md text-body-md shadow-sm focus:outline-none focus:bg-surface-tint-ice/50 transition-all placeholder:text-slate-body/50" id="full-name" placeholder="e.g. Divyansh Rajat" required type="text" />
          </div>
        </div>
        {/* Field 3: Organization / Department */}
        <div className="flex flex-col gap-1.5">
          <label className="font-label-lg text-label-lg text-on-surface flex items-center justify-between" htmlFor="organization">
            <span>Organization / Dept</span>
            <span className="font-label-code text-label-code text-tertiary text-[11px]">INSTITUTION</span>
          </label>
          <div className="relative flex items-center">
            <span className="material-symbols-outlined absolute left-3 text-slate-body text-[18px]">corporate_fare</span>
            <input value={organization} onChange={(e) => setOrganization(e.target.value)} className="w-full pl-10 pr-3.5 py-2.5 rounded-lg bg-surface-crisp text-on-surface font-body-md text-body-md shadow-sm focus:outline-none focus:bg-surface-tint-ice/50 transition-all placeholder:text-slate-body/50" id="organization" placeholder="e.g. Forensics Lab" required type="text" />
          </div>
        </div>
      </div>
      {/* Field 2: Work / Research Email */}
      <div className="flex flex-col gap-1.5">
        <label className="font-label-lg text-label-lg text-on-surface flex items-center justify-between" htmlFor="email">
          <span>Work / Research Email</span>
          <span className="font-label-code text-label-code text-primary text-[11px]">ENCRYPTED ROUTE</span>
        </label>
        <div className="relative flex items-center">
          <span className="material-symbols-outlined absolute left-3 text-slate-body text-[18px]">alternate_email</span>
          <input value={email} onChange={(e) => setEmail(e.target.value)} className="w-full pl-10 pr-3.5 py-2.5 rounded-lg bg-surface-crisp text-on-surface font-body-md text-body-md shadow-sm focus:outline-none focus:bg-surface-tint-ice/50 transition-all placeholder:text-slate-body/50" id="email" placeholder="analyst@forensics.org or university domain" required type="email" />
        </div>
      </div>
      {/* Password Section: Password & Confirm Password */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
        {/* Field 4: Password */}
        <div className="flex flex-col gap-1.5">
          <label className="font-label-lg text-label-lg text-on-surface flex items-center justify-between" htmlFor="password">
            <span>Security Key / Password</span>
          </label>
          <div className="relative flex items-center">
            <span className="material-symbols-outlined absolute left-3 text-slate-body text-[18px]">lock</span>
            <input value={password} onChange={(e) => setPassword(e.target.value)} className="w-full pl-10 pr-10 py-2.5 rounded-lg bg-surface-crisp text-on-surface font-body-md text-body-md shadow-sm focus:outline-none focus:bg-surface-tint-ice/50 transition-all" id="password" placeholder="••••••••••••" required type={showPassword ? 'text' : 'password'} />
            <button aria-label="Toggle password view" onClick={() => setShowPassword(!showPassword)} className="absolute right-2.5 text-slate-body hover:text-primary transition-colors flex items-center justify-center p-1" type="button">
              <span className="material-symbols-outlined text-[18px]">{showPassword ? 'visibility_off' : 'visibility'}</span>
            </button>
          </div>
        </div>
        {/* Field 5: Confirm Password */}
        <div className="flex flex-col gap-1.5">
          <label className="font-label-lg text-label-lg text-on-surface flex items-center justify-between" htmlFor="confirm-password">
            <span>Confirm Security Key</span>
            {password && confirmPassword && password === confirmPassword && (
              <span className="font-label-code text-[11px] text-verdict-real">MATCHED</span>
            )}
          </label>
          <div className="relative flex items-center">
            <span className="material-symbols-outlined absolute left-3 text-slate-body text-[18px]">check_circle</span>
            <input value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="w-full pl-10 pr-10 py-2.5 rounded-lg bg-surface-crisp text-on-surface font-body-md text-body-md shadow-sm focus:outline-none focus:bg-surface-tint-ice/50 transition-all" id="confirm-password" placeholder="••••••••••••" required type={showConfirmPassword ? 'text' : 'password'} />
            <button aria-label="Toggle confirm password view" onClick={() => setShowConfirmPassword(!showConfirmPassword)} className="absolute right-2.5 text-slate-body hover:text-primary transition-colors flex items-center justify-center p-1" type="button">
              <span className="material-symbols-outlined text-[18px]">{showConfirmPassword ? 'visibility_off' : 'visibility'}</span>
            </button>
          </div>
        </div>
      </div>
      
      {/* Real Live Password Strength Meter Panel */}
      <div className="p-space-sm rounded-xl bg-surface-tint-ice/80 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="material-symbols-outlined text-secondary text-[16px]">shield</span>
            <span className="font-label-lg text-[12px] text-slate-body uppercase tracking-wider">Entropy &amp; Strength:</span>
            <span className={`font-label-code text-label-code font-semibold ${strengthColor}`} id="strength-label">{strengthLabel}</span>
          </div>
        </div>
        {/* Segmented Colored Meter Gauge */}
        <div className="grid grid-cols-4 gap-1.5 h-1.5 w-full bg-surface-container rounded-full overflow-hidden p-0.5">
          <div className={`h-full rounded-full transition-all duration-300 ${barColors[0]}`} />
          <div className={`h-full rounded-full transition-all duration-300 ${barColors[1]}`} />
          <div className={`h-full rounded-full transition-all duration-300 ${barColors[2]}`} />
          <div className={`h-full rounded-full transition-all duration-300 ${barColors[3]}`} />
        </div>
        {/* Dynamic Requirements Checklist Chips */}
        <div className="grid grid-cols-3 gap-2 pt-1">
          <div className={`flex items-center gap-1 transition-colors ${hasLength ? 'text-emerald-500' : 'text-slate-body'}`}>
            <span className="material-symbols-outlined text-[14px]">
              {hasLength ? 'check_circle' : 'radio_button_unchecked'}
            </span>
            <span className="font-label-code text-[11px]">8+ Chars</span>
          </div>
          <div className={`flex items-center gap-1 transition-colors ${hasNum ? 'text-emerald-500' : 'text-slate-body'}`}>
            <span className="material-symbols-outlined text-[14px]">
              {hasNum ? 'check_circle' : 'radio_button_unchecked'}
            </span>
            <span className="font-label-code text-[11px]">1+ Number</span>
          </div>
          <div className={`flex items-center gap-1 transition-colors ${hasSym ? 'text-emerald-500' : 'text-slate-body'}`}>
            <span className="material-symbols-outlined text-[14px]">
              {hasSym ? 'check_circle' : 'radio_button_unchecked'}
            </span>
            <span className="font-label-code text-[11px]">Special Symbol</span>
          </div>
        </div>
      </div>
      {/* Decision Support Terms Checkbox */}
      <div className="flex items-start gap-space-sm pt-1">
        <label className="relative flex items-center cursor-pointer pt-0.5">
          <input defaultChecked className="w-4 h-4 rounded text-primary accent-primary focus:ring-0 focus:outline-none cursor-pointer" id="terms-check" required type="checkbox" />
        </label>
        <label className="font-body-sm text-body-sm text-slate-body leading-tight cursor-pointer select-none" htmlFor="terms-check">
          I agree to the Terms of Service and Privacy Policy.
        </label>
      </div>
      {/* Primary Action CTA Button */}
      <button disabled={isLoading} className="w-full py-3.5 px-6 rounded-full bg-primary hover:bg-electric-blue-hover text-on-primary font-label-lg text-label-lg font-semibold flex items-center justify-center gap-2 shadow-[0_4px_16px_rgba(0,80,203,0.35)] hover:shadow-[0_8px_24px_rgba(0,102,255,0.45)] hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 mt-2 cursor-pointer disabled:opacity-70 disabled:cursor-not-allowed" id="submit-btn" type="submit">
        <span>{isLoading ? 'Provisioning...' : 'Create Account & Launch Studio'}</span>
        <span className="material-symbols-outlined text-[19px]">rocket_launch</span>
      </button>
      {/* Switcher to Sign In Console */}
      <div className="flex items-center justify-center gap-1.5 pt-1 text-center">
        <span className="font-body-sm text-body-sm text-slate-body">Already registered?</span>
        <Link to="/login" className="font-label-lg text-body-sm text-primary hover:text-electric-blue-hover font-semibold transition-colors flex items-center gap-0.5">
          <span>Sign In to Console</span>
          <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
        </Link>
      </div>
    </form>
    {/* Success Feedback Overlay (Hidden by default) */}
    <div className="hidden absolute inset-0 bg-surface-crisp/95 backdrop-blur-md rounded-2xl flex flex-col items-center justify-center p-space-xl text-center z-20" id="signup-success">
      <div className="w-16 h-16 rounded-full bg-verdict-real-bg text-verdict-real flex items-center justify-center mb-space-md shadow-lg">
        <span className="material-symbols-outlined text-[36px]">verified</span>
      </div>
      <h2 className="font-headline-md text-headline-md text-on-surface">Analyst Identity Provisioned</h2>
      <p className="font-body-md text-body-md text-slate-body mt-2 max-w-sm">
        Cryptographic public credential has been registered. Initializing your multi-modal studio session...
      </p>
      <div className="mt-space-lg flex items-center gap-2 font-label-code text-label-code text-primary bg-surface-tint-ice px-4 py-2 rounded-full">
        <span className="w-2 h-2 rounded-full bg-cyan-energy animate-ping" />
        <span>REDIRECTING TO STUDIO SANDBOX (2s)</span>
      </div>
    </div>

  </div>

  {/* Live Password Strength & Interactive Micro-Script */}
</div>

    </div>
  );
}
