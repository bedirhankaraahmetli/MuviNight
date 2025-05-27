import React, { createContext, useContext, useState, useEffect } from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import { createAppTheme } from '../theme';

const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const [mode, setMode] = useState(() => {
    const savedMode = localStorage.getItem('themeMode');
    return savedMode || 'light';
  });

  const [themeName, setThemeName] = useState(() => {
    const savedTheme = localStorage.getItem('themeName');
    return savedTheme || 'default';
  });

  const [fontSize, setFontSize] = useState(() => {
    const savedFont = localStorage.getItem('fontSize');
    return savedFont || 'medium';
  });

  const [userPreferences, setUserPreferences] = useState(() => {
    const savedPrefs = localStorage.getItem('userPreferences');
    return savedPrefs ? JSON.parse(savedPrefs) : {
      fontSize: 'medium',
      reducedMotion: false,
      highContrast: false,
    };
  });

  useEffect(() => {
    localStorage.setItem('themeMode', mode);
  }, [mode]);

  useEffect(() => {
    localStorage.setItem('themeName', themeName);
  }, [themeName]);

  useEffect(() => {
    localStorage.setItem('fontSize', fontSize);
    document.body.style.fontSize = fontSize === 'small' ? '14px' : fontSize === 'large' ? '20px' : '16px';
  }, [fontSize]);

  useEffect(() => {
    localStorage.setItem('userPreferences', JSON.stringify(userPreferences));
  }, [userPreferences]);

  const toggleTheme = () => {
    setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
  };

  const updatePreferences = (newPreferences) => {
    setUserPreferences((prev) => ({ ...prev, ...newPreferences }));
  };

  const theme = createAppTheme(mode, themeName, fontSize);

  const value = {
    isDarkMode: mode === 'dark',
    mode,
    themeName,
    setThemeName,
    fontSize,
    setFontSize,
    toggleTheme,
    userPreferences,
    updatePreferences,
  };

  return (
    <ThemeContext.Provider value={value}>
      <MuiThemeProvider theme={theme}>
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
}; 