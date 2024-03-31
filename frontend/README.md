# Frontend

## Installation

Please install [NodeJS](https://nodejs.org/).

Please install VSCode extensions:

- Auto Rename Tag
- ES7+ React/Redux/React-Native snippets
- ESLint
- Highlight Matching Tag
- Prettier - Code formatter
- Tailwind CSS IntelliSense
- XML

To install dependencies, from `./frontend` folder run the command:

```bash
npm install
```

To update dependencies, from `./frontend` folder run the command:

```bash
npm update --save
```

To start implementing you need to run the development server:

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) with your browser to see the result.

## Backend endpoints

To generate TypeScript endpoint definition from the backend OpenAPI:

```bash
npx openapi-typescript http://localhost:8000/openapi.json --output ./src/v2/services/backend/endpoints.d.ts
```