import { BrowserRouter, Routes, Route, Link, NavLink, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ProtectedRoute } from './components/ProtectedRoute';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Upload from './pages/Upload';
import Dashboard from './pages/Dashboard';
import Report from './pages/Report';
import History from './pages/History';
import Transparency from './pages/Transparency';
import ForgotPassword from './pages/ForgotPassword';

const GlobalBackground = () => (
  <div className="fixed inset-0 pointer-events-none z-[-1] overflow-hidden bg-background">
    {/* Large ambient blur circles */}
    <div className="absolute top-[-20%] left-[-10%] w-[60vw] h-[60vw] max-w-[800px] max-h-[800px] bg-cyan-energy/10 rounded-full blur-3xl opacity-60" />
    <div className="absolute bottom-[-20%] right-[-10%] w-[60vw] h-[60vw] max-w-[800px] max-h-[800px] bg-primary/10 rounded-full blur-3xl opacity-60" />
    
    {/* Foreground Crosshairs / Focus Brackets */}
    <div className="absolute top-[10%] right-[20%] w-[40px] h-[40px] border-t-2 border-r-2 border-cyan-energy/20" />
    <div className="absolute bottom-[15%] left-[15%] w-[40px] h-[40px] border-b-2 border-l-2 border-primary/20" />
    
    {/* Forensic Metadata Streams */}
    <div className="absolute top-[25%] left-[8%] font-label-code text-[10px] text-slate-500/20 tracking-[0.3em] rotate-90 origin-left">
      HASH: 0x9F8B7A4C3D2E1F0 &nbsp;&nbsp;|&nbsp;&nbsp; NODE: CHK-01
    </div>
    <div className="absolute bottom-[20%] right-[8%] font-label-code text-[10px] text-slate-500/20 tracking-[0.3em] -rotate-90 origin-right">
      ATTESTATION: VERIFIED &nbsp;&nbsp;|&nbsp;&nbsp; SEQ: 104859
    </div>

    {/* Large Side Logos (Visible on empty sides of the screen) */}
    <div className="absolute inset-y-0 left-0 flex items-center justify-start pointer-events-none -translate-x-1/4">
      <img src="/logo.png" alt="" className="w-[50vw] max-w-[700px] object-contain opacity-[0.08] -rotate-12" />
    </div>
    <div className="absolute inset-y-0 right-0 flex items-center justify-end pointer-events-none translate-x-1/4">
      <img src="/logo.png" alt="" className="w-[50vw] max-w-[700px] object-contain opacity-[0.08] rotate-12" />
    </div>
    
    {/* Theme Words (Anchored to corners, no center alignment, avoiding navbar) */}
    <div className="absolute top-28 left-[4%] text-[3vw] md:text-[1.5vw] font-display font-black text-slate-500/10 whitespace-nowrap select-none tracking-widest">
      ZERO-TRUST
    </div>
    <div className="absolute bottom-[4%] right-[4%] text-[3vw] md:text-[1.5vw] font-display font-black text-slate-500/10 text-right select-none tracking-[0.2em] leading-tight">
      DEEPFAKE<br />DETECTION
    </div>
    
    {/* Cyber Grid pattern overlay */}
    <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMiIgY3k9IjIiIHI9IjEiIGZpbGw9InJnYmEoMCwwLDAsMC4wNSkiLz48L3N2Zz4=')] opacity-[0.35]" style={{ WebkitMaskImage: 'radial-gradient(circle at center, transparent 30%, black 100%)', maskImage: 'radial-gradient(circle at center, transparent 30%, black 100%)' }} />
  </div>
);

function Navigation() {
  const { user, isAuthenticated, logout } = useAuth();
  
  if (!isAuthenticated) {
    return (
      <header className="sticky top-0 w-full z-50 bg-surface-crisp/90 backdrop-blur-xl shadow-[0_1px_8px_rgba(11,25,44,0.04)]">
        <div className="w-full px-4 sm:px-6 h-20 flex items-center justify-between gap-space-md">
          <div className="flex items-center gap-space-md shrink-0">
            <Link to="/" className="flex items-center gap-space-md">
              <img alt="logo.png" className="h-8 w-auto object-contain" src="/logo.png" />
              <div className="flex flex-col">
                <div className="flex items-center gap-space-xs">
                  <span className="font-headline-sm text-headline-sm text-on-surface tracking-tight">CheckReality</span>
                  <span className="font-headline-sm text-headline-sm text-primary font-bold">AI</span>
                </div>
                <span className="font-label-caption text-label-caption text-secondary font-semibold uppercase tracking-wider">VERIFY • DETECT • STAY REAL</span>
              </div>
            </Link>
          </div>
          
          <div className="flex items-center gap-space-md shrink-0">
            <Link to="/login" className="px-space-md py-space-xs rounded-full font-label-lg text-label-lg text-on-surface-variant hover:text-on-surface transition-colors font-semibold">Login</Link>
            <Link to="/signup" className="px-space-lg py-space-xs rounded-full bg-primary text-on-primary font-label-lg text-label-lg font-semibold hover:bg-electric-blue-hover shadow-[0_4px_14px_rgba(0,102,255,0.3)] transition-all">Sign Up</Link>
          </div>
        </div>
      </header>
    );
  }

  const navLinkClass = ({ isActive }: { isActive: boolean }) => 
    isActive 
      ? "px-space-md py-space-xs transition-colors bg-surface-tint-ice text-primary font-bold rounded-full" 
      : "px-space-md py-space-xs rounded-full font-label-lg text-label-lg text-on-surface-variant hover:text-on-surface transition-colors";

  return (
    <header className="sticky top-0 w-full z-50 bg-surface-crisp/90 backdrop-blur-xl shadow-[0_1px_8px_rgba(11,25,44,0.04)]">
      <div className="w-full px-4 sm:px-6 h-20 flex items-center justify-between gap-space-md">
        <div className="flex items-center gap-space-md shrink-0">
          <Link to="/" className="flex items-center gap-space-md">
            <img alt="logo.png" className="h-8 w-auto object-contain" src="/logo.png" />
            <div className="flex flex-col">
              <div className="flex items-center gap-space-xs">
                <span className="font-headline-sm text-headline-sm text-on-surface tracking-tight">CheckReality</span>
                <span className="font-headline-sm text-headline-sm text-primary font-bold">AI</span>
              </div>
              <span className="font-label-caption text-label-caption text-secondary font-semibold uppercase tracking-wider">VERIFY • DETECT • STAY REAL</span>
            </div>
          </Link>
        </div>
        
        <nav className="hidden xl:flex items-center gap-space-xs">
          <NavLink to="/" className={navLinkClass} end>Overview</NavLink>
          <NavLink to="/upload" className={navLinkClass}>Studio</NavLink>
          <NavLink to="/dashboard" className={navLinkClass}>Fusion Engine</NavLink>
          <NavLink to="/history" className={navLinkClass}>History</NavLink>
          <NavLink to="/transparency" className={navLinkClass}>Transparency</NavLink>
        </nav>

        <div className="flex items-center gap-space-md shrink-0">
          <div className="hidden lg:flex items-center gap-space-xs">
            <div className="flex items-center gap-space-xs px-space-sm py-space-xs rounded-full bg-verdict-real-bg">
              <span className="w-2 h-2 rounded-full bg-verdict-real animate-pulse"></span>
              <span className="font-label-code text-label-code text-verdict-real font-semibold">SYSTEM: OPERATIONAL</span>
            </div>
            <div className="flex items-center gap-space-xs px-space-sm py-space-xs rounded-full bg-surface-tint-ice">
              <span className="w-2 h-2 rounded-full bg-cyan-energy"></span>
              <span className="font-label-code text-label-code text-secondary font-semibold">MODELS: 4 ACTIVE</span>
            </div>
          </div>
          
          <Link to="/upload" className="hidden sm:inline-flex items-center gap-space-xs px-space-lg py-space-xs rounded-full bg-primary-container text-on-primary font-label-lg text-label-lg font-semibold hover:bg-electric-blue-hover shadow-[0_4px_14px_rgba(0,102,255,0.3)] transition-all">
            <span className="material-symbols-outlined text-[18px]">csv</span>
            <span>Analyze Content</span>
          </Link>
          
          <div className="relative group flex items-center gap-3 ml-2 cursor-pointer p-1 pr-2 rounded-full hover:bg-surface-tint-ice transition-colors border border-transparent hover:border-outline/10" onClick={logout} title="Click to Logout">
            <div className="flex flex-col items-end hidden md:flex">
              <span className="font-bold text-sm text-on-surface leading-tight">{(user as any)?.name || (user as any)?.email?.split('@')[0] || 'User'}</span>
              <span className="font-label-code text-[10px] text-primary font-semibold leading-tight mt-0.5">{(user as any)?.email}</span>
            </div>
            <div className="w-9 h-9 rounded-full bg-primary flex items-center justify-center shrink-0 shadow-[0_2px_8px_rgba(0,102,255,0.25)]">
              <span className="material-symbols-outlined text-on-primary text-[20px]">person</span>
            </div>
            <div className="absolute right-0 top-12 w-32 py-2.5 bg-surface-crisp rounded-xl shadow-[0_4px_20px_rgba(0,0,0,0.1)] text-center text-sm font-bold text-red-500 hidden group-hover:block border border-outline/10 z-50">
              <span className="material-symbols-outlined text-[18px] align-text-bottom mr-1">logout</span>
              Logout
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function Footer() {
  const location = useLocation();
  if (location.pathname === '/') return null;

  return (
    <footer className="w-full mt-auto shrink-0 py-4 text-center flex flex-col items-center gap-1 z-30 bg-surface-crisp/80 backdrop-blur-md border-t border-outline/20 shadow-[0_-4px_20px_rgba(11,25,44,0.05)]">
      <div className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-surface-crisp/90 backdrop-blur-md shadow-sm font-label-code text-body-sm text-slate-body border border-outline/20">
        <span className="material-symbols-outlined text-primary text-[16px]">code_blocks</span>
        <span>Project Developed by <strong className="text-on-surface font-semibold text-primary">Divyansh Rajat</strong></span>
      </div>
      <p className="font-label-caption text-[11px] text-tertiary">
        CheckReality AI Multimodal Forensic Engine • Verified Zero-Trust Attestation
      </p>
    </footer>
  );
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <div className="min-h-screen flex flex-col bg-transparent relative z-10">
          <GlobalBackground />
          <Navigation />
          <main className="flex-grow">
            <Routes>
              {/* Public Routes */}
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<Login />} />
              <Route path="/signup" element={<Signup />} />
              <Route path="/forgot-password" element={<ForgotPassword />} />
              <Route path="/transparency" element={<Transparency />} />
              
              {/* Protected Routes */}
              <Route element={<ProtectedRoute />}>
                <Route path="/upload" element={<Upload />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/report/:id" element={<Report />} />
                <Route path="/history" element={<History />} />
              </Route>
            </Routes>
          </main>
          <Footer />
        </div>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
