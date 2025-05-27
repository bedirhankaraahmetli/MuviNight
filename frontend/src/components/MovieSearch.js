import React, { useState, useEffect, useRef } from 'react';
import Autocomplete from '@mui/material/Autocomplete';
import TextField from '@mui/material/TextField';
import InputAdornment from '@mui/material/InputAdornment';
import SearchIcon from '@mui/icons-material/Search';
import axios from 'axios';
import { useTheme } from '@mui/material/styles';

const API_BASE_URL = 'http://localhost:5000/api';

export default function MovieSearch({ selectedMovies, setSelectedMovies }) {
  const [movies, setMovies] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef();
  const theme = useTheme();

  useEffect(() => {
    if (!inputValue || inputValue.length < 2) {
      setMovies([]);
      return;
    }
    setLoading(true);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      axios
        .get(`${API_BASE_URL}/search`, { params: { query: inputValue, limit: 20 } })
        .then((res) => {
          let results;
          if (typeof res.data === 'string') {
            results = JSON.parse(res.data).results;
          } else {
            results = res.data.results;
          }
          setMovies(results || []);
        })
        .catch(() => setMovies([]))
        .finally(() => setLoading(false));
    }, 300);
    return () => clearTimeout(debounceRef.current);
  }, [inputValue]);

  // Debug: See what movies are being passed to Autocomplete
  console.log('Movies for Autocomplete:', movies);

  return (
    <Autocomplete
      options={movies}
      getOptionLabel={option => (option && option.title ? option.title : '')}
      value={null}
      onChange={(e, newValue) => {
        if (
          newValue &&
          !selectedMovies.find((m) => m.id === newValue.id) &&
          selectedMovies.length < 5
        ) {
          setSelectedMovies([...selectedMovies, newValue]);
        }
      }}
      inputValue={inputValue}
      onInputChange={(event, value) => setInputValue(value)}
      loading={loading}
      noOptionsText={
        !inputValue
          ? 'Start typing to search for movies, e.g., "The Godfather".'
          : 'No movies found matching your search.'
      }
      renderInput={(params) => (
        <TextField
          {...params}
          label="Search for movies..."
          variant="outlined"
          fullWidth
          sx={{
            background: theme.palette.background.paper,
            borderRadius: 2,
            maxWidth: 600,
            mx: 'auto',
            input: { color: theme.palette.text.primary, fontSize: '1.1rem', padding: '14px 18px' },
            label: { color: theme.palette.text.secondary },
            '& fieldset': { borderRadius: 2 },
          }}
          InputProps={{
            ...params.InputProps,
            endAdornment: (
              <InputAdornment position="end">
                <SearchIcon sx={{ color: '#a78bfa', fontSize: 28 }} />
              </InputAdornment>
            ),
          }}
        />
      )}
      disabled={selectedMovies.length >= 5}
      openOnFocus
      sx={{
        display: 'flex',
        justifyContent: 'center',
        mb: 4,
        borderRadius: 2,
        '& .MuiAutocomplete-paper': {
          borderRadius: 2,
        },
      }}
    />
  );
} 