import React, { useState, useEffect } from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Rating from '@mui/material/Rating';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Button from '@mui/material/Button';
import MovieCard from './MovieCard';
import axios from 'axios';
import ControlsBar from "./ControlsBar";
import MovieSearch from './MovieSearch';
import MovieFilters from './MovieFilters';
import Snackbar from '@mui/material/Snackbar';
import MuiAlert from '@mui/material/Alert';
import { useTheme } from '@mui/material/styles';
import MovieCreationIcon from '@mui/icons-material/MovieCreation';

const API_BASE_URL = 'http://localhost:5000/api';

function MovieDetailsModal({ open, movie, onClose }) {
  const theme = useTheme();
  if (!movie) return null;
  const posterUrl = movie.poster_path
    ? (movie.poster_path.startsWith('http') ? movie.poster_path : `https://image.tmdb.org/t/p/w500${movie.poster_path}`)
    : 'https://via.placeholder.com/300x450?text=No+Poster';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" PaperProps={{ sx: { background: theme.palette.background.paper, borderRadius: 3 } }}>
      <DialogTitle sx={{ color: theme.palette.text.primary, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
        <MovieCreationIcon sx={{ color: theme.palette.primary.main, fontSize: 28, mr: 1 }} />
        {movie.title}
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{ position: 'absolute', right: 16, top: 16, color: theme.palette.text.primary }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', p: 3 }}>
        <img
          src={posterUrl}
          alt={movie.title}
          style={{ width: 300, height: 450, objectFit: 'contain', borderRadius: 12, marginBottom: 24 }}
        />
        <Box sx={{ color: theme.palette.text.primary, textAlign: 'left', width: '100%' }}>
          <strong>{movie.title} {movie.release_date ? `(${new Date(movie.release_date).getFullYear()})` : ''}</strong>
          <p style={{ marginTop: 8 }}>{movie.overview}</p>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
            <Rating value={movie.vote_average / 2} precision={0.1} readOnly size="small" sx={{ color: '#FFD700' }} />
            <span style={{ color: '#FFD700', fontWeight: 600 }}>{movie.vote_average ? movie.vote_average.toFixed(1) : 'N/A'}/10</span>
          </Box>
        </Box>
      </Box>
    </Dialog>
  );
}

function MovieRecommender() {
  const [selectedMovies, setSelectedMovies] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalMovie, setModalMovie] = useState(null);
  const [sortBy, setSortBy] = useState("relevance");
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [filters, setFilters] = useState({
    genres: [],
    yearRange: [1900, new Date().getFullYear()],
    ratingRange: [0, 10],
    sortBy: 'relevance',
  });
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  const handleMovieRemove = (movieId) => {
    setSelectedMovies(selectedMovies.filter(movie => movie.id !== movieId));
  };

  const handleGetRecommendations = async () => {
    if (selectedMovies.length < 2) return;
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/recommend`, {
        movie_ids: selectedMovies.map(movie => movie.id),
        n_recommendations: 5
      });
      setRecommendations(response.data.recommendations);
    } catch (error) {
      alert('Error getting recommendations. Please try again.');
    }
    setLoading(false);
  };

  const handleSurpriseMe = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/surprise`);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      alert('Error getting surprise recommendations.');
    }
    setLoading(false);
  };

  const handleSortChange = (e) => setSortBy(e.target.value);
  const handleShare = () => {
    if (!recommendations.length) return;
    const text = recommendations.map(m => `${m.title} (${m.release_date?.split('-')[0] || ''})`).join('\n');
    navigator.clipboard.writeText(text);
    showSnackbar('Recommendations copied to clipboard!', 'success');
  };

  const handleFilter = () => setFilterModalOpen(true);

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
  };

  // Filter recommendations based on filters
  const filteredRecommendations = React.useMemo(() => {
    return recommendations.filter((movie) => {
      const year = movie.release_date ? new Date(movie.release_date).getFullYear() : 0;
      const rating = movie.vote_average || 0;
      const inYearRange = year >= filters.yearRange[0] && year <= filters.yearRange[1];
      const inRatingRange = rating >= filters.ratingRange[0] && rating <= filters.ratingRange[1];
      // If genres filter is empty, allow all genres
      const inGenres =
        !filters.genres.length ||
        (movie.genre_names && filters.genres.some((g) => movie.genre_names.includes(g)));
      return inYearRange && inRatingRange && inGenres;
    });
  }, [recommendations, filters]);

  const sortedRecommendations = React.useMemo(() => {
    let recs = [...filteredRecommendations];
    if (filters.sortBy === 'rating') {
      recs.sort((a, b) => b.vote_average - a.vote_average);
    } else if (filters.sortBy === 'year') {
      recs.sort((a, b) => {
        const ay = a.release_date ? new Date(a.release_date).getFullYear() : 0;
        const by = b.release_date ? new Date(b.release_date).getFullYear() : 0;
        return by - ay;
      });
    } else if (filters.sortBy === 'title') {
      recs.sort((a, b) => (a.title || '').localeCompare(b.title || '', undefined, { sensitivity: 'base' }));
    }
    // Default: relevance (original order)
    return recs;
  }, [filteredRecommendations, filters]);

  const showSnackbar = (message, severity = 'success') => {
    setSnackbar({ open: true, message, severity });
  };

  return (
    <Box>
      {/* Movie Selection */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Select Movies (2-5)
        </Typography>
        <MovieSearch
          selectedMovies={selectedMovies}
          setSelectedMovies={setSelectedMovies}
        />
        <Grid container spacing={2} sx={{ mt: 2 }}>
          {selectedMovies.map((movie) => (
            <Grid item xs={12} sm={6} md={4} key={movie.id}>
              <MovieCard
                movie={movie}
                onRemove={() => handleMovieRemove(movie.id)}
              />
            </Grid>
          ))}
        </Grid>
      </Paper>
      {/* Action Buttons */}
      <Box sx={{ mb: 3, display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={handleGetRecommendations}
          disabled={selectedMovies.length < 2 || loading}
        >
          Get Recommendations
        </Button>
        <Button
          variant="outlined"
          color="secondary"
          onClick={handleSurpriseMe}
          disabled={loading}
        >
          Surprise Me
        </Button>
      </Box>
      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Recommended Movies
          </Typography>
          <ControlsBar
            sortBy={filters.sortBy}
            onSortChange={(e) => setFilters(f => ({ ...f, sortBy: e.target.value }))}
            onShare={handleShare}
            onFilter={handleFilter}
          />
          <Grid container spacing={2}>
            {sortedRecommendations.map((movie) => (
              <Grid item xs={12} sm={6} md={4} key={movie.id}>
                <MovieCard
                  movie={movie}
                  onClick={() => {
                    setModalMovie(movie);
                    setModalOpen(true);
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}
      {/* Movie Filters Modal */}
      <Dialog open={filterModalOpen} onClose={() => setFilterModalOpen(false)} maxWidth="sm" fullWidth>
        <Box sx={{ p: 3 }}>
          <MovieFilters
            onFilterChange={handleFilterChange}
            onSortChange={(sortBy) => setFilters(f => ({ ...f, sortBy }))}
            onReset={() => setFilters({ genres: [], yearRange: [1900, new Date().getFullYear()], ratingRange: [0, 10], sortBy: 'relevance' })}
            genres={Array.from(new Set(recommendations.flatMap(m => m.genre_names || [])))}
            years={recommendations.map(m => m.release_date ? new Date(m.release_date).getFullYear() : 0)}
            initialFilters={filters}
          />
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button variant="contained" onClick={() => setFilterModalOpen(false)}>
              Apply Filters
            </Button>
          </Box>
        </Box>
      </Dialog>
      {/* Movie Details Modal */}
      <MovieDetailsModal
        open={modalOpen}
        movie={modalMovie}
        onClose={() => setModalOpen(false)}
      />
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar(s => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <MuiAlert elevation={6} variant="filled" severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </MuiAlert>
      </Snackbar>
    </Box>
  );
}

export default MovieRecommender; 