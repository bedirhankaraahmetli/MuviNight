import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  IconButton,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Rating,
  Stack,
  Chip,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import { useTheme } from '../context/ThemeContext';

const Watchlist = () => {
  const { isDarkMode } = useTheme();
  const [watchlist, setWatchlist] = useState(() => {
    const saved = localStorage.getItem('watchlist');
    return saved ? JSON.parse(saved) : [];
  });
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [userRating, setUserRating] = useState(0);
  const [userNotes, setUserNotes] = useState('');

  useEffect(() => {
    localStorage.setItem('watchlist', JSON.stringify(watchlist));
  }, [watchlist]);

  const handleAddToWatchlist = (movie) => {
    if (!watchlist.find(m => m.id === movie.id)) {
      setWatchlist([...watchlist, { ...movie, userRating: 0, userNotes: '' }]);
    }
  };

  const handleRemoveFromWatchlist = (movieId) => {
    setWatchlist(watchlist.filter(movie => movie.id !== movieId));
  };

  const handleEditClick = (movie) => {
    setSelectedMovie(movie);
    setUserRating(movie.userRating || 0);
    setUserNotes(movie.userNotes || '');
    setEditDialogOpen(true);
  };

  const handleSaveEdit = () => {
    setWatchlist(watchlist.map(movie =>
      movie.id === selectedMovie.id
        ? { ...movie, userRating, userNotes }
        : movie
    ));
    setEditDialogOpen(false);
  };

  const getPosterUrl = (movie) => {
    return movie.poster_path
      ? (movie.poster_path.startsWith('http')
        ? movie.poster_path
        : `https://image.tmdb.org/t/p/w500${movie.poster_path}`)
      : 'https://via.placeholder.com/300x450?text=No+Poster';
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ color: isDarkMode ? '#fff' : '#000' }}>
        My Watchlist
      </Typography>

      {watchlist.length === 0 ? (
        <Paper sx={{ p: 3, textAlign: 'center', background: isDarkMode ? '#2d2d44' : '#f5f5f5' }}>
          <Typography sx={{ color: isDarkMode ? '#aaa' : '#666' }}>
            Your watchlist is empty. Add movies to get started!
          </Typography>
        </Paper>
      ) : (
        <Grid container spacing={3}>
          {watchlist.map((movie) => (
            <Grid item xs={12} sm={6} md={4} key={movie.id}>
              <Paper
                sx={{
                  p: 2,
                  height: '100%',
                  background: isDarkMode ? '#2d2d44' : '#f5f5f5',
                  position: 'relative',
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
                  {/* User Rating */}
                  <Box>
                    <Typography
                      variant="subtitle2"
                      sx={{ color: isDarkMode ? '#aaa' : '#666' }}
                    >
                      Your Rating
                    </Typography>
                    <Rating
                      value={movie.userRating}
                      readOnly
                      size="small"
                      sx={{ color: '#FFD700' }}
                    />
                  </Box>

                  {/* User Notes */}
                  {movie.userNotes && (
                    <Box>
                      <Typography
                        variant="subtitle2"
                        sx={{ color: isDarkMode ? '#aaa' : '#666' }}
                      >
                        Your Notes
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ color: isDarkMode ? '#ddd' : '#333' }}
                      >
                        {movie.userNotes}
                      </Typography>
                    </Box>
                  )}

                  {/* Action Buttons */}
                  <Box sx={{ display: 'flex', gap: 1, mt: 'auto' }}>
                    <IconButton
                      onClick={() => handleEditClick(movie)}
                      size="small"
                      sx={{ color: isDarkMode ? '#fff' : '#000' }}
                    >
                      <EditIcon />
                    </IconButton>
                    <IconButton
                      onClick={() => handleRemoveFromWatchlist(movie.id)}
                      size="small"
                      sx={{ color: isDarkMode ? '#fff' : '#000' }}
                    >
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Stack>
              </Paper>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Edit Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        PaperProps={{
          sx: {
            background: isDarkMode ? '#23233a' : '#ffffff',
            borderRadius: 3,
          },
        }}
      >
        <DialogTitle sx={{ color: isDarkMode ? '#fff' : '#000' }}>
          Edit Movie Details
        </DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 2 }}>
            <Box>
              <Typography
                variant="subtitle2"
                sx={{ color: isDarkMode ? '#aaa' : '#666', mb: 1 }}
              >
                Your Rating
              </Typography>
              <Rating
                value={userRating}
                onChange={(event, newValue) => setUserRating(newValue)}
                precision={0.5}
                sx={{ color: '#FFD700' }}
              />
            </Box>
            <TextField
              label="Your Notes"
              multiline
              rows={4}
              value={userNotes}
              onChange={(e) => setUserNotes(e.target.value)}
              fullWidth
              sx={{
                '& .MuiOutlinedInput-root': {
                  color: isDarkMode ? '#fff' : '#000',
                  '& fieldset': {
                    borderColor: isDarkMode ? '#444' : '#ccc',
                  },
                },
                '& .MuiInputLabel-root': {
                  color: isDarkMode ? '#aaa' : '#666',
                },
              }}
            />
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setEditDialogOpen(false)}
            sx={{ color: isDarkMode ? '#fff' : '#000' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSaveEdit}
            variant="contained"
            sx={{ background: '#FFD700', color: '#000' }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Watchlist; 