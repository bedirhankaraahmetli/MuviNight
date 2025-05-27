import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  TextField,
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import Button from '@mui/material/Button';

const MovieFilters = ({
  onFilterChange,
  onSortChange,
  onReset,
  years,
  initialFilters = {
    yearRange: [1900, new Date().getFullYear()],
    ratingRange: [0, 10],
    sortBy: 'relevance',
  },
}) => {
  const [filters, setFilters] = useState(initialFilters);

  // Year Range Handlers
  const handleYearRangeChange = (event, newValue) => {
    setFilters(prev => ({ ...prev, yearRange: newValue }));
    onFilterChange({ ...filters, yearRange: newValue });
  };
  const handleYearInputChange = (idx, value) => {
    let newRange = [...filters.yearRange];
    newRange[idx] = value === '' ? '' : Number(value);
    setFilters(prev => ({ ...prev, yearRange: newRange }));
    onFilterChange({ ...filters, yearRange: newRange });
  };

  // Rating Range Handlers
  const handleRatingRangeChange = (event, newValue) => {
    setFilters(prev => ({ ...prev, ratingRange: newValue }));
    onFilterChange({ ...filters, ratingRange: newValue });
  };
  const handleRatingInputChange = (idx, value) => {
    let newRange = [...filters.ratingRange];
    newRange[idx] = value === '' ? '' : Number(value);
    setFilters(prev => ({ ...prev, ratingRange: newRange }));
    onFilterChange({ ...filters, ratingRange: newRange });
  };

  // Sort Handler
  const handleSortChange = (event) => {
    const newSortBy = event.target.value;
    setFilters(prev => ({ ...prev, sortBy: newSortBy }));
    onSortChange(newSortBy);
  };

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <FilterListIcon sx={{ mr: 1 }} />
        <Typography variant="h6">Filters & Sort</Typography>
      </Box>
      <Stack spacing={3}>
        {/* Year Range Filter */}
        <Box>
          <Typography gutterBottom>Year Range</Typography>
          <Slider
            value={filters.yearRange}
            onChange={handleYearRangeChange}
            valueLabelDisplay="auto"
            min={1900}
            max={new Date().getFullYear()}
          />
          <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
            <TextField
              label="Min Year"
              type="number"
              value={filters.yearRange[0]}
              onChange={e => handleYearInputChange(0, e.target.value)}
              inputProps={{ min: 1900, max: filters.yearRange[1] }}
              size="small"
            />
            <TextField
              label="Max Year"
              type="number"
              value={filters.yearRange[1]}
              onChange={e => handleYearInputChange(1, e.target.value)}
              inputProps={{ min: filters.yearRange[0], max: new Date().getFullYear() }}
              size="small"
            />
          </Box>
        </Box>

        {/* Rating Range Filter */}
        <Box>
          <Typography gutterBottom>Rating Range</Typography>
          <Slider
            value={filters.ratingRange}
            onChange={handleRatingRangeChange}
            valueLabelDisplay="auto"
            min={0}
            max={10}
            step={0.1}
          />
          <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
            <TextField
              label="Min Rating"
              type="number"
              value={filters.ratingRange[0]}
              onChange={e => handleRatingInputChange(0, e.target.value)}
              inputProps={{ min: 0, max: filters.ratingRange[1], step: 0.1 }}
              size="small"
            />
            <TextField
              label="Max Rating"
              type="number"
              value={filters.ratingRange[1]}
              onChange={e => handleRatingInputChange(1, e.target.value)}
              inputProps={{ min: filters.ratingRange[0], max: 10, step: 0.1 }}
              size="small"
            />
          </Box>
        </Box>

        {/* Sort Options */}
        <FormControl fullWidth>
          <InputLabel>Sort By</InputLabel>
          <Select
            value={filters.sortBy}
            onChange={handleSortChange}
            label="Sort By"
          >
            <MenuItem value="relevance">Relevance</MenuItem>
            <MenuItem value="rating">Rating</MenuItem>
            <MenuItem value="year">Year</MenuItem>
            <MenuItem value="title">Title</MenuItem>
          </Select>
        </FormControl>
      </Stack>
      <Box sx={{ display: 'flex', gap: 2, mt: 2, justifyContent: 'flex-end' }}>
        <Button variant="outlined" color="secondary" onClick={() => { setFilters(initialFilters); onReset && onReset(); }}>
          Reset Filters
        </Button>
      </Box>
    </Paper>
  );
};

export default MovieFilters; 