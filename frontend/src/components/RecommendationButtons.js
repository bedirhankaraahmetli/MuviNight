import React from 'react';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export default function RecommendationButtons({ selectedMovies, setRecommendations, setLoading, loading }) {
  const handleGetRecommendations = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE_URL}/recommend`, {
        movie_ids: selectedMovies.map(m => m.id),
        n_recommendations: 5,
        model_name: 'content_based'
      });
      console.log('Recommendations response:', res.data);
      setRecommendations(res.data.recommendations || []);
    } catch (e) {
      console.error('Error getting recommendations:', e);
      setRecommendations([]);
      alert('Failed to get recommendations. Please check the backend.');
    }
    setLoading(false);
  };

  const handleSurpriseMe = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE_URL}/surprise`);
      setRecommendations(res.data.recommendations || []);
    } catch (e) {
      setRecommendations([]);
    }
    setLoading(false);
  };

  return (
    <Stack direction="row" spacing={2} justifyContent="center">
      <Button
        variant="contained"
        color="primary"
        onClick={handleGetRecommendations}
        disabled={selectedMovies.length < 2 || loading}
        sx={{ minWidth: 180, fontWeight: 600 }}
      >
        Get Recommendations
      </Button>
      <Button
        variant="outlined"
        color="secondary"
        onClick={handleSurpriseMe}
        disabled={loading}
        sx={{ minWidth: 140, fontWeight: 600 }}
      >
        Surprise Me
      </Button>
    </Stack>
  );
} 