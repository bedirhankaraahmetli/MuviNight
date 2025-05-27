import React from 'react';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Rating from '@mui/material/Rating';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import { useTheme, alpha } from '@mui/material/styles';

const POSTER_BASE_URL = 'https://image.tmdb.org/t/p/w342';

export default function MovieCard({ movie, onClick, onRemove }) {
  const theme = useTheme();
  const { title, overview, poster_path, release_date, vote_average } = movie;
  const posterUrl = poster_path
    ? poster_path.startsWith('http')
      ? poster_path
      : `${POSTER_BASE_URL}${poster_path}`
    : 'https://via.placeholder.com/260x370?text=No+Poster';

  // Slightly darken the card background compared to the main background
  const cardBg = theme.palette.mode === 'light'
    ? alpha(theme.palette.background.paper, 0.95)
    : alpha(theme.palette.background.paper, 0.85);

  return (
    <Card
      sx={{
        width: 260,
        minWidth: 260,
        background: cardBg,
        borderRadius: 3,
        boxShadow: 3,
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'stretch',
        cursor: onClick ? 'pointer' : 'default',
        transition: 'box-shadow 0.2s',
        '&:hover': { boxShadow: 8 },
        zIndex: 10,
      }}
      onClick={onClick}
    >
      {/* Remove Button */}
      {onRemove && (
        <IconButton
          aria-label="remove"
          onClick={e => {
            e.stopPropagation();
            onRemove();
          }}
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            color: '#fff',
            background: 'rgba(30,30,30,0.7)',
            '&:hover': { background: 'rgba(60,60,60,0.9)' },
            zIndex: 20,
          }}
          size="small"
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      )}
      <CardMedia
        component="img"
        height="370"
        image={posterUrl}
        alt={title}
        sx={{ objectFit: 'cover', borderTopLeftRadius: 12, borderTopRightRadius: 12 }}
      />
      <CardContent sx={{ p: 2, pb: 2, background: cardBg, borderBottomLeftRadius: 12, borderBottomRightRadius: 12 }}>
        <Typography gutterBottom variant="subtitle1" noWrap sx={{ color: theme.palette.text.primary, fontWeight: 700, fontSize: '1.08rem', mb: 1 }}>
          {title}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
          <Typography variant="body2" sx={{ color: theme.palette.text.secondary, fontWeight: 500 }}>
            {release_date ? new Date(release_date).getFullYear() : ''}
          </Typography>
          <Rating
            value={vote_average / 2}
            precision={0.1}
            readOnly
            size="small"
            sx={{ color: '#FFD700' }}
          />
          <Typography variant="body2" sx={{ color: theme.palette.text.secondary, fontWeight: 500 }}>
            {vote_average ? vote_average.toFixed(1) : 'N/A'}
          </Typography>
        </Box>
        {overview && (
          <Typography
            variant="body2"
            sx={{
              color: theme.palette.text.secondary,
              mt: 1,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
            }}
          >
            {overview}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
} 