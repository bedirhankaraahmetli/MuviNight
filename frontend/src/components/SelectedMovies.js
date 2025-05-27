import React, { useState } from 'react';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardMedia from '@mui/material/CardMedia';
import CardContent from '@mui/material/CardContent';
import Rating from '@mui/material/Rating';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';

const POSTER_BASE_URL = 'https://image.tmdb.org/t/p/w780';

export default function SelectedMovies({ selectedMovies, setSelectedMovies }) {
  const [open, setOpen] = useState(false);
  const [modalMovie, setModalMovie] = useState(null);

  const handleCardClick = (movie) => {
    setModalMovie(movie);
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setModalMovie(null);
  };

  return (
    <Box sx={{ mb: 3, maxWidth: 1500, mx: 'auto' }}>
      <Typography variant="h6" sx={{ color: '#fff', mb: 2, fontWeight: 700 }}>
        Selected Movies
      </Typography>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'row',
          gap: 2,
          justifyContent: 'center',
          flexWrap: 'wrap',
        }}
      >
        {selectedMovies.map(movie => (
          <Card
            key={movie.id}
            sx={{
              width: 260,
              minWidth: 260,
              background: '#23233a',
              borderRadius: 3,
              boxShadow: 3,
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'stretch',
              cursor: 'pointer',
              transition: 'box-shadow 0.2s',
              '&:hover': { boxShadow: 8 }
            }}
            onClick={() => handleCardClick(movie)}
          >
            <IconButton
              sx={{
                position: 'absolute',
                top: 8,
                right: 8,
                zIndex: 2,
                background: 'rgba(0,0,0,0.5)',
                color: '#fff',
                '&:hover': { background: 'rgba(167,139,250,0.9)' }
              }}
              onClick={e => {
                e.stopPropagation();
                setSelectedMovies(selectedMovies.filter(m => m.id !== movie.id));
              }}
            >
              <CloseIcon fontSize="medium" />
            </IconButton>
            <CardMedia
              component="img"
              height="370"
              image={
                movie.poster_path
                  ? movie.poster_path.startsWith('http')
                    ? movie.poster_path
                    : `${POSTER_BASE_URL}${movie.poster_path}`
                  : 'https://via.placeholder.com/260x370?text=No+Poster'
              }
              alt={movie.title}
              sx={{ objectFit: 'cover', borderTopLeftRadius: 12, borderTopRightRadius: 12 }}
            />
            <CardContent sx={{ p: 2, pb: 2, background: '#23233a', borderBottomLeftRadius: 12, borderBottomRightRadius: 12 }}>
              <Typography gutterBottom variant="subtitle1" noWrap sx={{ color: '#fff', fontWeight: 700, fontSize: '1.08rem', mb: 1 }}>
                {movie.title}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                <Typography variant="body2" color="#bdbdbd" sx={{ fontWeight: 500 }}>
                  {movie.release_date ? new Date(movie.release_date).getFullYear() : ''}
                </Typography>
                <Rating
                  value={movie.vote_average / 2}
                  precision={0.1}
                  readOnly
                  size="small"
                  sx={{ color: '#FFD700' }}
                />
                <Typography variant="body2" color="#bdbdbd" sx={{ fontWeight: 500 }}>
                  {movie.vote_average ? movie.vote_average.toFixed(1) : 'N/A'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        ))}
      </Box>

      {/* Movie Details Modal */}
      <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
        {modalMovie && (
          <>
            <DialogTitle sx={{ fontWeight: 700, fontSize: '1.3rem', background: '#35354a', color: '#fff' }}>
              {modalMovie.title}
              <IconButton
                aria-label="close"
                onClick={handleClose}
                sx={{
                  position: 'absolute',
                  right: 16,
                  top: 16,
                  color: '#fff',
                  background: 'rgba(0,0,0,0.2)'
                }}
              >
                <CloseIcon />
              </IconButton>
            </DialogTitle>
            <DialogContent sx={{ background: '#35354a', color: '#fff', p: 3 }}>
              <Box sx={{ mb: 2 }}>
                <img
                  src={
                    modalMovie.poster_path
                      ? (modalMovie.poster_path.startsWith('http')
                        ? modalMovie.poster_path
                        : `https://image.tmdb.org/t/p/w500${modalMovie.poster_path}`)
                      : (modalMovie.backdrop_path
                        ? (modalMovie.backdrop_path.startsWith('http')
                          ? modalMovie.backdrop_path
                          : `https://image.tmdb.org/t/p/w780${modalMovie.backdrop_path}`)
                        : 'https://via.placeholder.com/340x500?text=No+Image')
                  }
                  alt={modalMovie.title}
                  style={{
                    width: '100%',
                    maxHeight: 500,
                    objectFit: 'contain',
                    borderRadius: 12,
                    background: '#23233a',
                    marginBottom: 16
                  }}
                />
              </Box>
              <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                {modalMovie.title} {modalMovie.release_date ? `(${new Date(modalMovie.release_date).getFullYear()})` : ''}
              </Typography>
              <Typography variant="body1" sx={{ mb: 2 }}>
                {modalMovie.overview}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Rating
                  value={modalMovie.vote_average / 2}
                  precision={0.1}
                  readOnly
                  size="medium"
                  sx={{ color: '#FFD700' }}
                />
                <Typography variant="body2" sx={{ color: '#bdbdbd', fontWeight: 600 }}>
                  {modalMovie.vote_average ? modalMovie.vote_average.toFixed(1) : 'N/A'}/10
                </Typography>
                {modalMovie.genres && (
                  <Typography variant="body2" sx={{ color: '#bdbdbd', ml: 2 }}>
                    <b>Genres:</b> {Array.isArray(modalMovie.genres) ? modalMovie.genres.join(', ') : modalMovie.genres}
                  </Typography>
                )}
              </Box>
            </DialogContent>
          </>
        )}
      </Dialog>
    </Box>
  );
} 