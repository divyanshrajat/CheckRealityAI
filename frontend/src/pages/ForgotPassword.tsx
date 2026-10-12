import React, { useState } from 'react';
import { Link } from 'react-router-dom';

export default function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    // Simulate network request
    setTimeout(() => {
      setIsLoading(false);
      setIsSubmitted(true);
    }, 1200);
  };

  return (
    <div className="w-full flex flex-col">
      <div className="flex flex-col w-full relative min-h-[calc(100vh-4rem)] items-center justify-center p-4 sm:p-6 lg:p-8 overflow-hidden">
        
        {/* Dynamic Holographic Underlay */}
        <div className="absolute inset-0 pointer-events-none flex items-center justify-center -z-10 overflow-hidden">
          <div className="absolute -top-36 -left-36 w-[550px] h-[550px] bg-gradient-to-br from-cyan-energy/15 via-primary/10 to-transparent rounded-full blur-3xl transform -rotate-12" />
          <div className="absolute -bottom-40 -right-40 w-[640px] h-[640px] bg-gradient-to-tl from-secondary-container/20 via-primary-container/10 to-transparent rounded-full blur-3xl" />
        </div>

        {/* Forgot Password Card */}
        <div className="w-full max-w-md relative z-10 animate-fade-in-up">
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-surface-tint-ice to-transparent opacity-80 blur-lg -z-10" />
          <div className="bg-surface-crisp/90 backdrop-blur-2xl rounded-3xl p-space-2xl sm:p-space-3xl shadow-[0_8px_32px_rgba(11,25,44,0.06)] border border-slate-border/40 flex flex-col">
            
            <div className="flex flex-col items-center mb-space-xl text-center">
              <div className="w-16 h-16 rounded-2xl bg-surface-tint-ice flex items-center justify-center shadow-sm mb-space-sm">
                <span className="material-symbols-outlined text-[32px] text-primary">lock_reset</span>
              </div>
              <h1 className="font-display text-display-sm text-on-surface font-bold tracking-tight mb-2">Reset Password</h1>
              <p className="font-body-md text-body-md text-slate-body">
                Enter the email associated with your account and we'll send you a link to reset your password.
              </p>
            </div>

            {isSubmitted ? (
              <div className="flex flex-col items-center text-center animate-fade-in">
                <div className="w-16 h-16 rounded-full bg-verdict-real-bg text-verdict-real flex items-center justify-center mb-space-md">
                  <span className="material-symbols-outlined text-[32px]">mark_email_read</span>
                </div>
                <h3 className="font-headline-sm text-headline-sm text-on-surface font-bold mb-2">Check your inbox</h3>
                <p className="font-body-md text-body-md text-slate-body mb-space-xl">
                  If an account exists for <span className="font-semibold text-primary">{email}</span>, we've sent instructions for resetting your password.
                </p>
                <Link to="/login" className="w-full py-3 px-6 rounded-full bg-surface-container text-on-surface font-label-lg text-label-lg font-semibold hover:bg-surface-tint-ice transition-all flex items-center justify-center">
                  Return to Login
                </Link>
              </div>
            ) : (
              <form className="flex flex-col gap-space-lg" onSubmit={handleSubmit}>
                <div className="flex flex-col gap-1.5">
                  <label className="font-label-md text-label-md text-on-surface-variant font-medium ml-1" htmlFor="reset-email">Email Address</label>
                  <div className="relative group">
                    <span className="absolute left-4 top-1/2 -translate-y-1/2 material-symbols-outlined text-outline text-[20px] group-focus-within:text-primary transition-colors">alternate_email</span>
                    <input 
                      autoComplete="email" 
                      className="w-full pl-11 pr-4 py-3.5 bg-surface-container-low text-on-surface border border-slate-border/60 rounded-xl focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all placeholder:text-outline-variant font-body-md" 
                      id="reset-email" 
                      placeholder="analyst@agency.gov" 
                      required 
                      type="email" 
                      value={email} 
                      onChange={(e) => setEmail(e.target.value)} 
                    />
                  </div>
                </div>

                <div className="pt-2">
                  <button disabled={isLoading} className="relative w-full py-3.5 px-6 rounded-full bg-gradient-to-r from-primary to-primary-container hover:from-electric-blue-hover hover:to-primary text-on-primary font-label-lg text-label-lg tracking-wide shadow-[0_4px_18px_rgba(0,80,203,0.35)] hover:shadow-[0_6px_24px_rgba(0,216,246,0.4)] flex items-center justify-center transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed" type="submit">
                    {isLoading ? (
                      <span className="flex items-center gap-2">
                        <span className="material-symbols-outlined animate-spin text-[18px]">sync</span>
                        Sending Link...
                      </span>
                    ) : (
                      <span className="flex items-center gap-2">
                        Send Reset Link
                        <span className="material-symbols-outlined text-[18px]">arrow_forward</span>
                      </span>
                    )}
                  </button>
                </div>
                
                <div className="flex items-center justify-center pt-2 text-center">
                  <span className="font-body-md text-body-md text-slate-body">
                    Remember your password?{' '}
                    <Link className="text-primary hover:text-electric-blue-hover font-semibold transition-colors" to="/login">
                      Sign In
                    </Link>
                  </span>
                </div>
              </form>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
