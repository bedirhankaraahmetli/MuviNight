import React from "react";
import { Box, Button, Select, MenuItem, Typography } from "@mui/material";
import ShareIcon from "@mui/icons-material/Share";
import FilterListIcon from "@mui/icons-material/FilterList";
import { useTheme } from '@mui/material/styles';

const sortOptions = [
  { value: "relevance", label: "Relevance" },
  { value: "rating", label: "Rating" },
  { value: "year", label: "Year" },
  { value: "title", label: "Title" },
];

export default function ControlsBar({
  sortBy,
  onSortChange,
  onShare,
  onFilter,
}) {
  const theme = useTheme();
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 2,
        mt: 2,
        mb: 3,
        flexWrap: "wrap",
      }}
    >
      <Button
        variant="outlined"
        startIcon={<ShareIcon />}
        sx={{
          color: theme.palette.primary.main,
          borderColor: theme.palette.primary.main,
          background: theme.palette.background.paper,
          '&:hover': {
            borderColor: theme.palette.primary.dark,
            background: theme.palette.action.hover,
          },
          fontWeight: 600,
          letterSpacing: 1,
        }}
        onClick={onShare}
      >
        SHARE
      </Button>
      <Typography sx={{ color: theme.palette.text.secondary, fontWeight: 500, mx: 1 }}>
        Sort By
      </Typography>
      <Select
        value={sortBy}
        onChange={onSortChange}
        sx={{
          minWidth: 140,
          color: theme.palette.primary.contrastText,
          borderRadius: 2,
          background: theme.palette.primary.main,
          border: `1.5px solid ${theme.palette.primary.main}`,
          fontWeight: 600,
          '& .MuiSelect-icon': { color: theme.palette.primary.light },
        }}
        MenuProps={{
          PaperProps: {
            sx: { background: theme.palette.primary.main, color: theme.palette.primary.contrastText },
          },
        }}
      >
        {sortOptions.map((opt) => (
          <MenuItem key={opt.value} value={opt.value}>
            {opt.label}
          </MenuItem>
        ))}
      </Select>
      <Button
        variant="outlined"
        startIcon={<FilterListIcon />}
        sx={{
          color: theme.palette.primary.main,
          borderColor: theme.palette.primary.main,
          background: theme.palette.background.paper,
          '&:hover': {
            borderColor: theme.palette.primary.dark,
            background: theme.palette.action.hover,
          },
          fontWeight: 600,
          letterSpacing: 1,
        }}
        onClick={onFilter}
      >
        FILTERS
      </Button>
    </Box>
  );
} 