import React, { useState, useEffect } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import {
  Box,
  CssBaseline,
  Container,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  useMediaQuery,
  useTheme as useMuiTheme,
  Fade,
  Grow,
  Tooltip,
} from '@mui/material';
import {
  Brightness4,
  Brightness7,
  Menu as MenuIcon,
  Movie as MovieIcon,
  List as ListIcon,
  Compare as CompareIcon,
  Settings as SettingsIcon,
  Palette as PaletteIcon,
  MovieCreation as MovieCreationIcon,
  InfoOutlined as InfoOutlinedIcon,
} from '@mui/icons-material';
import MovieRecommender from './components/MovieRecommender';
import Watchlist from './components/Watchlist';
import MovieComparison from './components/MovieComparison';
import { useTheme } from './context/ThemeContext';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Switch from '@mui/material/Switch';
import Menu from '@mui/material/Menu';
import { themePalettes } from './theme';

const drawerWidth = 280;

function AppContent() {
  const { themeName, setThemeName, fontSize, setFontSize, isDarkMode, toggleTheme } = useTheme();
  const muiTheme = useMuiTheme();
  const isMobile = useMediaQuery(muiTheme.breakpoints.down('sm'));
  const [currentView, setCurrentView] = useState('recommender');
  const [comparisonMovies, setComparisonMovies] = useState([]);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);

  useEffect(() => {
    document.body.style.fontSize = fontSize === 'small' ? '14px' : fontSize === 'large' ? '20px' : '16px';
  }, [fontSize]);

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: '100%',
          ml: 0,
          backdropFilter: 'blur(8px)',
          backgroundColor: 'background.paper',
        }}
      >
        <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <MovieCreationIcon sx={{ fontSize: 44, color: muiTheme.palette.primary.main, mr: 1 }} />
            <Typography
              variant="h4"
              noWrap
              component="div"
              sx={{
                fontWeight: 900,
                fontSize: { xs: '2rem', sm: '2.5rem', md: '2.8rem' },
                background: `linear-gradient(90deg, ${muiTheme.palette.primary.main} 0%, ${muiTheme.palette.primary.light} 50%, ${muiTheme.palette.secondary.main} 100%)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
                letterSpacing: 1,
              }}
            >
              MuviNight
            </Typography>
          </Box>
          <Tooltip title="How to use MuviNight?">
            <IconButton
              onClick={() => setInfoOpen(true)}
              sx={{
                bgcolor: 'background.paper',
                color: 'primary.main',
                border: '1px solid #bbb',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                mr: 2,
                '&:hover': {
                  bgcolor: 'background.default',
                },
              }}
            >
              <InfoOutlinedIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Settings">
            <IconButton
              onClick={() => setSettingsOpen(true)}
              sx={{
                bgcolor: 'background.paper',
                color: 'primary.main',
                border: '1px solid #bbb',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                mr: 2,
                '&:hover': {
                  bgcolor: 'background.default',
                },
              }}
            >
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: '100%',
          minHeight: '100vh',
          background: 'background.paper',
          transition: 'background-color 0.3s ease-in-out',
        }}
      >
        <Toolbar />
        <Container maxWidth="lg">
          <Fade in timeout={500}>
            <Box>
              <MovieRecommender />
            </Box>
          </Fade>
        </Container>
      </Box>

      <MovieComparison
        open={comparisonOpen}
        movies={comparisonMovies}
        onClose={() => setComparisonOpen(false)}
      />

      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>User Customization</DialogTitle>
        <DialogContent>
          <Box sx={{ my: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 2 }}>
              <Typography sx={{ minWidth: 120 }}>Dark Mode</Typography>
              <Switch checked={isDarkMode} onChange={toggleTheme} />
              <Typography>{isDarkMode ? 'On' : 'Off'}</Typography>
            </Box>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel id="theme-label">Color Theme</InputLabel>
              <Select
                labelId="theme-label"
                value={themeName}
                label="Color Theme"
                onChange={e => setThemeName(e.target.value)}
              >
                {Object.keys(themePalettes).map(key => (
                  <MenuItem key={key} value={key}>
                    {key.charAt(0).toUpperCase() + key.slice(1)}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel id="font-size-label">Font Size</InputLabel>
              <Select
                labelId="font-size-label"
                value={fontSize}
                label="Font Size"
                onChange={e => setFontSize(e.target.value)}
              >
                <MenuItem value="small">Small</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="large">Large</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)} variant="contained">Close</Button>
        </DialogActions>
      </Dialog>

      <Dialog open={infoOpen} onClose={() => setInfoOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>How to use MuviNight?</DialogTitle>
        <DialogContent>
          <Box sx={{ my: 2 }}>
            <Typography variant="h6" gutterBottom>Welcome to MuviNight!</Typography>
            <Typography variant="body1" gutterBottom>
              <b>1. Search for Movies:</b> Use the search bar to find movies by title. Start typing and select from the suggestions.
            </Typography>
            <Typography variant="body1" gutterBottom>
              <b>2. Select Movies:</b> Click on movies to add them to your selection (2-5 movies).
            </Typography>
            <Typography variant="body1" gutterBottom>
              <b>3. Get Recommendations:</b> Click "Get Recommendations" to see movies you might like based on your selection.
            </Typography>
            <Typography variant="body1" gutterBottom>
              <b>4. Use Filters & Sorting:</b> Use the Filters button to filter recommendations by year, rating, and more. Sort results as you like.
            </Typography>
            <Typography variant="body1" gutterBottom>
              <b>5. Share:</b> Use the Share button to copy your recommendations to the clipboard.
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Enjoy discovering new movies!
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setInfoOpen(false)} variant="contained">Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;
