import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Typography,
  IconButton,
  Snackbar,
  Alert,
  Stack,
  Chip,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import ShareIcon from '@mui/icons-material/Share';
import { useTheme } from '../context/ThemeContext';

const ShareMovie = ({ open, movie, onClose }) => {
  const { isDarkMode } = useTheme();
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  const handleShare = async () => {
    try {
      // Here you would typically make an API call to send the email
      // For now, we'll just show a success message
      setSnackbar({
        open: true,
        message: 'Movie shared successfully!',
        severity: 'success',
      });
      onClose();
    } catch (error) {
      setSnackbar({
        open: true,
        message: 'Failed to share movie. Please try again.',
        severity: 'error',
      });
    }
  };

  const handleCopyLink = () => {
    const shareUrl = `${window.location.origin}/movie/${movie.id}`;
    navigator.clipboard.writeText(shareUrl);
    setSnackbar({
      open: true,
      message: 'Link copied to clipboard!',
      severity: 'success',
    });
  };

  const handleShareSocial = (platform) => {
    const shareUrl = `${window.location.origin}/movie/${movie.id}`;
    const shareText = `Check out ${movie.title} on MuviNight!`;
    
    let shareLink;
    switch (platform) {
      case 'twitter':
        shareLink = `https://twitter.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`;
        break;
      case 'facebook':
        shareLink = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`;
        break;
      case 'linkedin':
        shareLink = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
        break;
      default:
        return;
    }
    
    window.open(shareLink, '_blank', 'width=600,height=400');
  };

  return (
    <>
      <Dialog
        open={open}
        onClose={onClose}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            background: isDarkMode ? '#23233a' : '#ffffff',
            borderRadius: 3,
          },
        }}
      >
        <DialogTitle sx={{ color: isDarkMode ? '#fff' : '#000' }}>
          Share Movie
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
          <Stack spacing={3} sx={{ mt: 2 }}>
            {/* Movie Info */}
            <Box>
              <Typography
                variant="h6"
                sx={{ color: isDarkMode ? '#fff' : '#000', mb: 1 }}
              >
                {movie.title}
              </Typography>
              {movie.release_date && (
                <Typography
                  variant="subtitle1"
                  sx={{ color: isDarkMode ? '#aaa' : '#666' }}
                >
                  {new Date(movie.release_date).getFullYear()}
                </Typography>
              )}
            </Box>

            {/* Share via Email */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{ color: isDarkMode ? '#aaa' : '#666', mb: 1 }}
              >
                Share via Email
              </Typography>
              <TextField
                label="Email Address"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                fullWidth
                sx={{
                  mb: 2,
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
              <TextField
                label="Message (optional)"
                multiline
                rows={3}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
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
            </Box>

            {/* Share via Link */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{ color: isDarkMode ? '#aaa' : '#666', mb: 1 }}
              >
                Share via Link
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  startIcon={<ContentCopyIcon />}
                  onClick={handleCopyLink}
                  sx={{
                    color: isDarkMode ? '#fff' : '#000',
                    borderColor: isDarkMode ? '#444' : '#ccc',
                  }}
                >
                  Copy Link
                </Button>
              </Box>
            </Box>

            {/* Social Media Share */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{ color: isDarkMode ? '#aaa' : '#666', mb: 1 }}
              >
                Share on Social Media
              </Typography>
              <Stack direction="row" spacing={1}>
                <Chip
                  label="Twitter"
                  onClick={() => handleShareSocial('twitter')}
                  sx={{
                    background: '#1DA1F2',
                    color: '#fff',
                    '&:hover': { background: '#1a8cd8' },
                  }}
                />
                <Chip
                  label="Facebook"
                  onClick={() => handleShareSocial('facebook')}
                  sx={{
                    background: '#4267B2',
                    color: '#fff',
                    '&:hover': { background: '#365899' },
                  }}
                />
                <Chip
                  label="LinkedIn"
                  onClick={() => handleShareSocial('linkedin')}
                  sx={{
                    background: '#0077B5',
                    color: '#fff',
                    '&:hover': { background: '#006699' },
                  }}
                />
              </Stack>
            </Box>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={onClose}
            sx={{ color: isDarkMode ? '#fff' : '#000' }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleShare}
            variant="contained"
            startIcon={<ShareIcon />}
            disabled={!email}
            sx={{ background: '#FFD700', color: '#000' }}
          >
            Share
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default ShareMovie; 