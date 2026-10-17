import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService } from '../api/auth';

interface User {
  id: string;
  email: string;
  full_name: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (token: string, user: User) => void;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const savedUser = localStorage.getItem('user');
      if (!savedUser || savedUser === 'undefined' || savedUser === 'null') return null;
      return JSON.parse(savedUser);
    } catch (e) {
      localStorage.removeItem('user');
      return null;
    }
  });
  const [token, setToken] = useState<string | null>(() => {
    const stored = localStorage.getItem('token');
    return (stored && stored !== 'undefined' && stored !== 'null') ? stored : null;
  });
  const [isLoading, setIsLoading] = useState(true);

  const login = (newToken: string, newUser: any) => {
    if (!newToken || typeof newToken !== 'string' || newToken === 'undefined') return;
    const userObj = (newUser && typeof newUser === 'object')
      ? newUser
      : { id: 'temp', email: '', full_name: 'Analyst' };
    localStorage.setItem('token', newToken);
    localStorage.setItem('user', JSON.stringify(userObj));
    setToken(newToken);
    setUser(userObj);
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
  };

  const checkAuth = async () => {
    const storedToken = localStorage.getItem('token');
    if (!storedToken || storedToken === 'undefined' || storedToken === 'null') {
      setIsLoading(false);
      return;
    }
    
    try {
      const userData = await authService.getMe();
      if (userData) {
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
      }
    } catch (err) {
      logout();
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkAuth();
    
    // Listen for unauthorized interceptor
    const handleUnauthorized = () => logout();
    window.addEventListener('auth:unauthorized', handleUnauthorized);
    
    return () => {
      window.removeEventListener('auth:unauthorized', handleUnauthorized);
    };
  }, []);

  return (
    <AuthContext.Provider value={{ user, token, isAuthenticated: !!token, isLoading, login, logout, checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
