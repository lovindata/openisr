import _ from "./app/page";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { Routes, Route, BrowserRouter } from "react-router-dom";

function App() {
  return (
    <main>
      <BrowserRouter>
        <QueryClientProvider client={new QueryClient()}>
          <Routes>
            <Route path="/app" element={<_ />} />
          </Routes>
        </QueryClientProvider>
      </BrowserRouter>
    </main>
  );
}

export default App;