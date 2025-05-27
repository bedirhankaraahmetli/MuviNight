import { createTheme } from '@mui/material/styles';

export const themePalettes = {
  default: {
    primary: { main: '#6366f1', light: '#818cf8', dark: '#4f46e5', contrastText: '#fff' },
    secondary: { main: '#f59e0b', light: '#fbbf24', dark: '#d97706', contrastText: '#000' },
    background: { default: '#f8fafc', paper: '#fff', card: '#f1f5f9' },
    dark: {
      primary: { main: '#a78bfa', light: '#c4b5fd', dark: '#7c3aed', contrastText: '#fff' },
      secondary: { main: '#fbbf24', light: '#fcd34d', dark: '#d97706', contrastText: '#000' },
      background: { default: '#0f172a', paper: '#1e293b', card: '#334155' },
    },
  },
  popcorn: {
    primary: { main: '#f9ca24', light: '#ffeaa7', dark: '#f6e58d', contrastText: '#222' },
    secondary: { main: '#e17055', light: '#fab1a0', dark: '#d35400', contrastText: '#fff' },
    background: { default: '#fffbe6', paper: '#fffbe6', card: '#fff9c4' },
    dark: {
      primary: { main: '#f6e58d', light: '#fffbe6', dark: '#f9ca24', contrastText: '#222' },
      secondary: { main: '#e17055', light: '#fab1a0', dark: '#d35400', contrastText: '#fff' },
      background: { default: '#222', paper: '#2d2d2d', card: '#333' },
    },
  },
  classic: {
    primary: { main: '#c0392b', light: '#e17055', dark: '#96281B', contrastText: '#fff' },
    secondary: { main: '#f6e58d', light: '#fffbe6', dark: '#f9ca24', contrastText: '#222' },
    background: { default: '#fff5f5', paper: '#fff5f5', card: '#ffeaea' },
    dark: {
      primary: { main: '#96281B', light: '#c0392b', dark: '#6C0A0A', contrastText: '#fff' },
      secondary: { main: '#f6e58d', light: '#fffbe6', dark: '#f9ca24', contrastText: '#222' },
      background: { default: '#1a1a1a', paper: '#2d2d2d', card: '#333' },
    },
  },
  ocean: {
    primary: { main: '#00bcd4', light: '#4dd0e1', dark: '#008394', contrastText: '#fff' },
    secondary: { main: '#009688', light: '#52c7b8', dark: '#00675b', contrastText: '#fff' },
    background: { default: '#e0f7fa', paper: '#b2ebf2', card: '#b2ebf2' },
    dark: {
      primary: { main: '#4dd0e1', light: '#00bcd4', dark: '#008394', contrastText: '#fff' },
      secondary: { main: '#52c7b8', light: '#009688', dark: '#00675b', contrastText: '#fff' },
      background: { default: '#102027', paper: '#37474f', card: '#263238' },
    },
  },
  sunset: {
    primary: { main: '#ff7675', light: '#fab1a0', dark: '#d63031', contrastText: '#fff' },
    secondary: { main: '#fdcb6e', light: '#ffeaa7', dark: '#e17055', contrastText: '#222' },
    background: { default: '#fff6e6', paper: '#fff1e6', card: '#ffeaa7' },
    dark: {
      primary: { main: '#d63031', light: '#ff7675', dark: '#c0392b', contrastText: '#fff' },
      secondary: { main: '#fdcb6e', light: '#ffeaa7', dark: '#e17055', contrastText: '#222' },
      background: { default: '#2d1a1a', paper: '#3e2723', card: '#4e342e' },
    },
  },
};

export const createAppTheme = (mode = 'light', themeName = 'default', fontSize = 'medium') => {
  const isDark = mode === 'dark';
  const paletteSet = themePalettes[themeName] || themePalettes.default;
  const palette = isDark ? paletteSet.dark : paletteSet;

  let baseFontSize = 16;
  if (fontSize === 'small') baseFontSize = 14;
  if (fontSize === 'large') baseFontSize = 20;

  return createTheme({
    palette: {
      mode,
      primary: palette.primary,
      secondary: palette.secondary,
      background: palette.background,
      text: {
        primary: isDark ? '#f1f5f9' : '#0f172a',
        secondary: isDark ? '#94a3b8' : '#475569',
      },
    },
    typography: {
      fontFamily: [
        'Inter',
        '-apple-system',
        'BlinkMacSystemFont',
        '"Segoe UI"',
        'Roboto',
        '"Helvetica Neue"',
        'Arial',
        'sans-serif',
      ].join(','),
      fontSize: baseFontSize,
      h1: { fontWeight: 700, fontSize: `${2.5 * baseFontSize / 16}rem` },
      h2: { fontWeight: 700, fontSize: `${2.0 * baseFontSize / 16}rem` },
      h3: { fontWeight: 600, fontSize: `${1.75 * baseFontSize / 16}rem` },
      h4: { fontWeight: 600, fontSize: `${1.5 * baseFontSize / 16}rem` },
      h5: { fontWeight: 600, fontSize: `${1.25 * baseFontSize / 16}rem` },
      h6: { fontWeight: 600, fontSize: `${1.0 * baseFontSize / 16}rem` },
    },
    shape: {
      borderRadius: 12,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 600,
            borderRadius: 8,
            padding: '8px 16px',
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              transform: 'translateY(-2px)',
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            },
          },
          contained: {
            boxShadow: 'none',
            '&:hover': {
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 16,
            transition: 'all 0.3s ease-in-out',
            '&:hover': {
              transform: 'translateY(-4px)',
              boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
            },
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
          elevation1: {
            boxShadow: isDark
              ? '0 2px 8px rgba(0,0,0,0.2)'
              : '0 2px 8px rgba(0,0,0,0.05)',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            borderRight: 'none',
            boxShadow: isDark
              ? '2px 0 8px rgba(0,0,0,0.2)'
              : '2px 0 8px rgba(0,0,0,0.05)',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            boxShadow: 'none',
            borderBottom: `1px solid ${isDark ? '#334155' : '#e2e8f0'}`,
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              transform: 'scale(1.05)',
            },
          },
        },
      },
      MuiIconButton: {
        styleOverrides: {
          root: {
            transition: 'all 0.2s ease-in-out',
            '&:hover': {
              transform: 'scale(1.1)',
            },
          },
        },
      },
    },
  });
}; 