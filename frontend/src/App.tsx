import { ImagePg } from "@/features/images/components/pages/ImagePg";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { Routes, Route, BrowserRouter, Navigate } from "react-router-dom";

function App() {
  return (
    <main>
      <BrowserRouter>
        <QueryClientProvider client={new QueryClient()}>
          <Routes>
            <Route path="/" element={<ImagePg />} />
            {<Route path="*" element={<Navigate to="/" />} />}
          </Routes>
        </QueryClientProvider>
      </BrowserRouter>
    </main>
  );
}

export default App;
