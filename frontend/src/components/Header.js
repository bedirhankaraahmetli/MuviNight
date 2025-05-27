import React from 'react';
import Typography from '@mui/material/Typography';
import MovieIcon from '@mui/icons-material/Movie';
import Box from '@mui/material/Box';

export default function Header() {
  return (
    <Box sx={{ textAlign: 'center', mb: 4 }}>
      <MovieIcon sx={{ fontSize: 64, color: '#a78bfa', mb: 1 }} />
      <Typography variant="h2" sx={{ fontWeight: 700, letterSpacing: 1, color: '#a78bfa', mb: 1 }}>
        MuviNight
      </Typography>
      <Typography variant="h6" sx={{ color: '#bdbdbd', mb: 2 }}>
        Discover your next favorite movie
      </Typography>
    </Box>
  );
} 