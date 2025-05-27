# MuviNight Frontend

This is the frontend application for the MuviNight movie recommendation system. Built with React, it provides a modern and responsive user interface for interacting with the recommendation system.

## Features

- Modern, responsive design
- Movie poster previews and detailed information
- Support for selecting 2-5 movies for personalized recommendations
- "Surprise Me" feature for random recommendations
- User preference management
- Real-time movie recommendations

## Prerequisites

- Node.js 14+
- npm 6+

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

The application will be available at [http://localhost:3000](http://localhost:3000).

## Available Scripts

### `npm start`

Runs the app in development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

### `npm test`

Launches the test runner in interactive watch mode.

### `npm run build`

Builds the app for production to the `build` folder.\
The build is optimized for the best performance.

## Project Structure

```
frontend/
├── public/              # Static files
├── src/                 # Source code
│   ├── components/     # React components
│   ├── pages/         # Page components
│   ├── services/      # API services
│   ├── utils/         # Utility functions
│   ├── App.js         # Main App component
│   └── index.js       # Entry point
├── package.json        # Dependencies and scripts
└── README.md          # This file
```

## API Integration

The frontend communicates with the backend API endpoints:

- `POST /api/recommend` - Get movie recommendations
- `GET /api/movies` - Get list of movies
- `GET /api/movies/<id>` - Get movie details
- `GET /api/surprise` - Get random recommendations

## Development

- The app uses React 18
- Styling is done with CSS modules
- API calls are handled using Axios
- State management is handled with React Context

## Building for Production

To create a production build:

```bash
npm run build
```

This will create an optimized build in the `build` folder.

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request

## License

This project is licensed under the MIT License.
