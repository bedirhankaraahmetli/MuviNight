import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import MovieCard from './MovieCard';
import CircularProgress from '@mui/material/CircularProgress';
import Button from '@mui/material/Button';
import FilterListIcon from '@mui/icons-material/FilterList';
import ShareIcon from '@mui/icons-material/Share';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';

export default function Recommendations({ recommendations, loading }) {
  const [sortBy, setSortBy] = React.useState('Relevance');

  if (loading) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <CircularProgress color="primary" />
      </Box>
    );
  }
  if (!recommendations || recommendations.length === 0) return null;

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h6" sx={{ color: '#fff', mb: 2, fontWeight: 700 }}>
        Recommended Movies
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Button variant="outlined" color="primary" startIcon={<ShareIcon />} sx={{ borderRadius: 2, fontWeight: 500 }}>
          Share
        </Button>
        <Typography sx={{ color: '#bdbdbd', fontSize: '1rem' }}>Sort By</Typography>
        <Select
          value={sortBy}
          onChange={e => setSortBy(e.target.value)}
          sx={{
            background: '#23233a',
            color: '#fff',
            borderRadius: 2,
            fontWeight: 500,
            '.MuiOutlinedInput-notchedOutline': { borderColor: '#a78bfa' },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': { borderColor: '#a78bfa' },
            '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: '#a78bfa' },
            minWidth: 120,
          }}
        >
          <MenuItem value="Relevance">Relevance</MenuItem>
          <MenuItem value="Year">Year</MenuItem>
          <MenuItem value="Rating">Rating</MenuItem>
        </Select>
        <Button variant="outlined" color="primary" startIcon={<FilterListIcon />} sx={{ borderRadius: 2, fontWeight: 500 }}>
          Filters
        </Button>
      </Box>
      <Grid container spacing={3} justifyContent="flex-start">
        {recommendations.map(movie => (
          <Grid item key={movie.id}>
            <MovieCard movie={movie} />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
} 