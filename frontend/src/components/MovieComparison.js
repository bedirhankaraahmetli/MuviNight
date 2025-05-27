import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Grid,
  Paper,
  Typography,
  Box,
  Rating,
  Chip,
  Stack,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useTheme } from '../context/ThemeContext';

const MovieComparison = ({ open, movies, onClose }) => {
  const { isDarkMode } = useTheme();

  if (!movies || movies.length === 0) return null;

  const getPosterUrl = (movie) => {
    return movie.poster_path
      ? (movie.poster_path.startsWith('http')
        ? movie.poster_path
        : `https://image.tmdb.org/t/p/w500${movie.poster_path}`)
      : 'https://via.placeholder.com/300x450?text=No+Poster';
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          background: isDarkMode ? '#23233a' : '#ffffff',
          borderRadius: 3,
        },
      }}
    >
      <DialogTitle sx={{ color: isDarkMode ? '#fff' : '#000' }}>
        Movie Comparison
        <IconButton
          aria-label="close"
          onClick={onClose}
          sx={{
            position: 'absolute',
            right: 16,
            top: 16,
            color: isDarkMode ? '#fff' : '#000',
          }}
        >
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          {movies.map((movie) => (
            <Grid item xs={12} md={6} lg={4} key={movie.id}>
              <Paper
                sx={{
                  p: 2,
                  height: '100%',
                  background: isDarkMode ? '#2d2d44' : '#f5f5f5',
                }}
              >
                <Box sx={{ textAlign: 'center', mb: 2 }}>
                  <img
                    src={getPosterUrl(movie)}
                    alt={movie.title}
                    style={{
                      width: '100%',
                      maxWidth: 300,
                      height: 'auto',
                      borderRadius: 8,
                    }}
                  />
                </Box>
                <Typography
                  variant="h6"
                  gutterBottom
                  sx={{ color: isDarkMode ? '#fff' : '#000' }}
                >
                  {movie.title}
                  {movie.release_date && (
                    <Typography
                      component="span"
                      variant="subtitle1"
                      sx={{ ml: 1, color: isDarkMode ? '#aaa' : '#666' }}
                    >
                      ({new Date(movie.release_date).getFullYear()})
                    </Typography>
                  )}
                </Typography>

                <Stack spacing={2}>
                  {/* Rating */}
                  <Box>
                    <Typography
                      variant="subtitle2"
                      sx={{ color: isDarkMode ? '#aaa' : '#666' }}
                    >
                      Rating
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Rating
                        value={movie.vote_average / 2}
                        precision={0.1}
                        readOnly
                        size="small"
                        sx={{ color: '#FFD700' }}
                      />
                      <Typography
                        sx={{ color: isDarkMode ? '#fff' : '#000' }}
                      >
                        {movie.vote_average.toFixed(1)}/10
                      </Typography>
                    </Box>
                  </Box>

                  {/* Genres */}
                  {movie.genres && (
                    <Box>
                      <Typography
                        variant="subtitle2"
                        sx={{ color: isDarkMode ? '#aaa' : '#666' }}
                      >
                        Genres
                      </Typography>
                      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                        {movie.genres.map((genre) => (
                          <Chip
                            key={genre}
                            label={genre}
                            size="small"
                            sx={{
                              background: isDarkMode ? '#3d3d5c' : '#e0e0e0',
                              color: isDarkMode ? '#fff' : '#000',
                            }}
                          />
                        ))}
                      </Stack>
                    </Box>
                  )}

                  {/* Overview */}
                  <Box>
                    <Typography
                      variant="subtitle2"
                      sx={{ color: isDarkMode ? '#aaa' : '#666' }}
                    >
                      Overview
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{ color: isDarkMode ? '#ddd' : '#333' }}
                    >
                      {movie.overview}
                    </Typography>
                  </Box>
                </Stack>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </DialogContent>
    </Dialog>
  );
};

export default MovieComparison; 