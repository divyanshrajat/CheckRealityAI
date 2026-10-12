// @ts-nocheck
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { authService } from '../api/auth';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    try {
      const res = await authService.login(email, password);
      login(res.access_token, res.user || { id: 'temp', email: email, full_name: 'Analyst' });
      navigate('/');
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail) || 'Failed to login');
    } finally {
      setIsLoading(false);
    }
  };
  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col w-full relative min-h-[calc(100vh-4rem)] items-center justify-center p-4 sm:p-6 lg:p-8 overflow-hidden">
  {/* Dynamic Luminous Holographic & Circuit Mesh Underlay (Inspired by Inspiration Art) */}
  <div className="absolute inset-0 pointer-events-none flex items-center justify-center -z-10 overflow-hidden">
    {/* Fluid Energy Waves & Radial Glows */}
    <div className="absolute -top-36 -left-36 w-[550px] h-[550px] bg-gradient-to-br from-cyan-energy/15 via-primary/10 to-transparent rounded-full blur-3xl transform -rotate-12" />
    <div className="absolute -bottom-40 -right-40 w-[640px] h-[640px] bg-gradient-to-tl from-secondary-container/20 via-primary-container/10 to-transparent rounded-full blur-3xl" />
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[900px] h-[700px] bg-radial from-surface-tint-ice/60 via-transparent to-transparent blur-2xl" />
    {/* Decorative Forensics SVG Circuit & Node Schematic Canvas */}
    <svg className="absolute w-full h-full text-slate-border/70 stroke-current opacity-40 max-w-7xl" fill="none" viewBox="0 0 1200 800" xmlns="http://www.w3.org/2000/svg">
      {/* Neural constellation connectors */}
      <path d="M120 220 Q240 180 380 290 T640 250 T920 380" stroke="url(#lineGrad1)" strokeDasharray="4 6" strokeWidth="1.5" />
      <path d="M180 620 C340 540 520 660 760 520 S1080 580 1140 480" stroke="url(#lineGrad2)" strokeWidth="1.5" />
      <circle cx={380} cy={290} fill="#00D8F6" r={4} />
      <circle cx={640} cy={250} fill="#0066FF" r={3} />
      <circle cx={760} cy={520} fill="#00D8F6" r={5} />
      <circle cx={920} cy={380} fill="#0050CB" r={4} />
      {/* Hexagonal lattice nodes */}
      <polygon fill="none" opacity="0.6" points="1060,160 1085,175 1085,205 1060,220 1035,205 1035,175" stroke="#CBD5E1" strokeWidth="1.2" />
      <polygon fill="none" opacity="0.8" points="1095,210 1120,225 1120,255 1095,270 1070,255 1070,225" stroke="#BAE6FD" strokeWidth="1.2" />
      <polygon fill="none" opacity="0.5" points="140,440 160,452 160,476 140,488 120,476 120,452" stroke="#CBD5E1" strokeWidth="1.2" />
      {/* Gradients for dynamic cyber glow paths */}
      <defs>
        <linearGradient id="lineGrad1" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#00D8F6" stopOpacity="0.2" />
          <stop offset="50%" stopColor="#0066FF" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#0050CB" stopOpacity="0.1" />
        </linearGradient>
        <linearGradient id="lineGrad2" x1="0%" x2="100%" y1="0%" y2="100%">
          <stop offset="0%" stopColor="#0050CB" stopOpacity="0.1" />
          <stop offset="60%" stopColor="#00D8F6" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#00C1FD" stopOpacity="0.3" />
        </linearGradient>
      </defs>
    </svg>
  </div>

  {/* Central High-Precision Glassmorphism Authentication Card */}
  <div className="relative w-full max-w-[540px] rounded-3xl bg-surface-crisp/95 backdrop-blur-2xl shadow-[0_20px_60px_-15px_rgba(0,80,203,0.12),0_4px_24px_rgba(0,216,246,0.06)] p-6 sm:p-10 transition-all duration-300">
    {/* Top Ambient Glow Line */}
    <div className="absolute top-0 inset-x-8 h-1 bg-gradient-to-r from-transparent via-cyan-energy to-transparent rounded-full opacity-70" />
    {/* Header & Verification Metaphor Header */}
    <div className="flex flex-col items-center text-center space-y-3">
      {/* High-Tech Shielded Logo Badge */}
      <div className="relative group">
        <div className="absolute -inset-2 bg-gradient-to-r from-cyan-energy/30 via-primary/20 to-secondary-container/30 rounded-2xl blur-lg group-hover:blur-xl transition duration-500 opacity-80" />
        <div className="relative w-16 h-16 rounded-2xl bg-surface-crisp flex items-center justify-center shadow-[0_4px_16px_rgba(0,80,203,0.1)] p-2">
          <img alt="CheckReality AI Logo" className="w-full h-full object-contain drop-shadow-sm" src="/logo.png" />
        </div>
        <div className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full bg-primary flex items-center justify-center text-on-primary shadow-sm">
          <span className="material-symbols-outlined text-[12px]">verified</span>
        </div>
      </div>
      {/* Text Headings */}
      <div className="space-y-1 pt-1">
        <h1 className="font-headline-lg text-headline-lg text-on-surface tracking-tight">
          Welcome Back to <span className="text-primary font-headline-lg text-headline-lg">CheckReality AI</span>
        </h1>
        <p className="font-body-md text-body-md text-on-surface-variant max-w-sm mx-auto">
          Verify, detect, and analyze synthetic media with multimodal neural precision.
        </p>
      </div>
      {/* Multimodal Support Micro Badges */}
      <div className="flex items-center justify-center gap-2 pt-1">
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface-tint-ice text-primary font-label-caption text-label-caption shadow-xs">
          <span className="material-symbols-outlined text-[13px]">description</span> Text
        </span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface-tint-ice text-primary font-label-caption text-label-caption shadow-xs">
          <span className="material-symbols-outlined text-[13px]">image</span> Image
        </span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface-tint-ice text-primary font-label-caption text-label-caption shadow-xs">
          <span className="material-symbols-outlined text-[13px]">videocam</span> Video
        </span>
        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface-tint-ice text-primary font-label-caption text-label-caption shadow-xs">
          <span className="material-symbols-outlined text-[13px]">graphic_eq</span> Audio
        </span>
      </div>
    </div>
    {/* Login Form */}
    {error && (
      <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 font-label-md text-label-md text-center">
        {error}
      </div>
    )}

    {/* Demo Access Credentials with Autofill */}
    <div className="mt-6 p-4 rounded-xl bg-primary-container/30 border border-primary/20 text-center">
      <p className="font-label-md text-label-md text-on-surface font-semibold mb-1">Demo Access Credentials</p>
      <div className="flex flex-col gap-1 items-center justify-center">
        <p className="font-label-code text-label-caption text-on-surface-variant">Email: <span className="font-bold text-primary">demo@checkreality.ai</span></p>
        <p className="font-label-code text-label-caption text-on-surface-variant">Password: <span className="font-bold text-primary">demo123</span></p>
      </div>
      <button 
        type="button" 
        onClick={() => { setEmail('demo@checkreality.ai'); setPassword('demo123'); }}
        className="mt-3 px-4 py-1.5 rounded-full bg-primary text-on-primary font-label-caption text-label-caption font-bold hover:bg-electric-blue-hover transition-colors shadow-sm mx-auto flex items-center gap-1"
      >
        <span className="material-symbols-outlined text-[14px]">magic_button</span>
        Autofill Credentials
      </button>
    </div>

    <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
      {/* Email Field */}
      <div className="space-y-1.5 text-left">
        <label className="block font-label-lg text-label-lg text-on-surface" htmlFor="email-address">
          Institutional / Analyst Email
        </label>
        <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-outline group-focus-within:text-primary transition-colors">
            <span className="material-symbols-outlined text-[20px]">alternate_email</span>
          </div>
          <input autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full pl-10 pr-4 py-3 bg-surface-tint-card hover:bg-surface-crisp focus:bg-surface-crisp text-on-surface font-body-md text-body-md rounded-xl shadow-xs transition-all duration-200 outline-none focus:shadow-[0_0_0_3px_rgba(0,216,246,0.25),0_1px_8px_rgba(0,80,203,0.12)] placeholder:text-outline-variant" id="email-address" name="email" placeholder="analyst@organization.com" required type="email" />
        </div>
      </div>
      {/* Password Field */}
      <div className="space-y-1.5 text-left">
        <div className="flex items-center justify-between">
          <label className="block font-label-lg text-label-lg text-on-surface" htmlFor="master-password">
            Master Security Key / Password
          </label>
          <span className="font-label-caption text-label-caption text-secondary">PKI/AES-256</span>
        </div>
        <div className="relative group">
          <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-outline group-focus-within:text-primary transition-colors">
            <span className="material-symbols-outlined text-[20px]">lock</span>
          </div>
          <input autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full pl-10 pr-12 py-3 bg-surface-tint-card hover:bg-surface-crisp focus:bg-surface-crisp text-on-surface font-body-md text-body-md rounded-xl shadow-xs transition-all duration-200 outline-none focus:shadow-[0_0_0_3px_rgba(0,216,246,0.25),0_1px_8px_rgba(0,80,203,0.12)] placeholder:text-outline-variant tracking-wider" id="master-password" name="password" placeholder="••••••••••••" required type={showPassword ? "text" : "password"} />
          <button onClick={() => setShowPassword(!showPassword)} className="absolute inset-y-0 right-0 pr-3.5 flex items-center text-outline hover:text-on-surface transition-colors focus:outline-none" id="toggle-pwd-btn" title="Toggle password visibility" type="button">
            <span className="material-symbols-outlined text-[20px]" id="toggle-pwd-icon">{showPassword ? "visibility_off" : "visibility"}</span>
          </button>
        </div>
      </div>
      {/* Remember & Forgot Options Row */}
      <div className="flex items-center justify-between pt-1">
        <label className="relative flex items-center gap-2.5 cursor-pointer select-none">
          <input defaultChecked className="peer sr-only" id="remember-me" type="checkbox" />
          <div className="w-5 h-5 rounded-md bg-surface-tint-card peer-checked:bg-primary transition-all duration-200 flex items-center justify-center shadow-xs peer-checked:shadow-[0_0_8px_rgba(0,80,203,0.3)]">
            <span className="material-symbols-outlined text-surface-crisp text-[14px] opacity-0 peer-checked:opacity-100 transition-opacity font-bold">check</span>
          </div>
          <span className="font-body-sm text-body-sm text-slate-body">Remember for 30 days</span>
        </label>
        <Link className="font-label-lg text-label-lg text-primary hover:text-electric-blue-hover transition-colors" to="/forgot-password">
          Forgot Password?
        </Link>
      </div>
      {/* Primary Action CTA Button */}
      <div className="pt-2">
        <button disabled={isLoading} className="relative w-full py-3.5 px-6 rounded-full bg-gradient-to-r from-primary to-primary-container hover:from-electric-blue-hover hover:to-primary text-on-primary font-label-lg text-label-lg tracking-wide shadow-[0_4px_18px_rgba(0,80,203,0.35)] hover:shadow-[0_6px_24px_rgba(0,216,246,0.4)] flex items-center justify-center gap-2 transition-all duration-300 transform active:scale-[0.99] group overflow-hidden disabled:opacity-70 disabled:cursor-not-allowed" id="submit-auth-btn" type="submit">
          <div className="absolute inset-0 w-1/2 h-full bg-surface-crisp/15 skew-x-12 -translate-x-full group-hover:translate-x-[300%] transition-transform duration-1000 ease-out pointer-events-none" />
          
          {!isLoading ? (
            <span className="flex items-center gap-2 font-label-lg text-label-lg" id="btn-text">
              Sign In to Verification Console
              <span className="material-symbols-outlined text-[18px] group-hover:translate-x-1 transition-transform">arrow_forward</span>
            </span>
          ) : (
            <span className="flex items-center gap-2" id="btn-spinner">
              <svg className="animate-spin h-5 w-5 text-on-primary" fill="none" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <circle className="opacity-25" cx={12} cy={12} r={10} stroke="currentColor" strokeWidth={4} />
                <path className="opacity-75" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" fill="currentColor" />
              </svg>
              <span className="font-label-lg text-label-lg">Authenticating Enclave...</span>
            </span>
          )}
        </button>
      </div>
    </form>
    {/* SSO Action Grid Removed */}
    {/* Security & Provenance Technical Trust Badge */}
    <div className="mt-8 pt-5 bg-surface-tint-ice/50 rounded-2xl p-3 text-center">
      <div className="flex items-center justify-center gap-1.5 text-primary mb-1">
        <span className="material-symbols-outlined text-[15px]">enhanced_encryption</span>
        <span className="font-label-code text-label-code tracking-tight font-bold">CRYPTO-SECURED ENCLAVE</span>
      </div>
      <p className="font-label-caption text-label-caption text-tertiary">
        Protected by Zero-Knowledge Enclave • 256-bit Encryption • Developed by Divyansh Rajat
      </p>
    </div>
    {/* Account Switcher Link */}
    <div className="mt-6 text-center">
      <p className="font-body-md text-body-md text-on-surface-variant">
        Don't have an analyst account?
        <Link to="/signup" className="font-label-lg text-label-lg text-primary hover:text-electric-blue-hover font-semibold transition-colors underline-offset-4 hover:underline ml-1">
          Create Account
        </Link>
      </p>
    </div>
  </div>
  {/* Interactive Client Behavior Script */}
</div>

    </div>
  );
}
