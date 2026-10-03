import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import ReportViewer from "./pages/ReportViewer";
import BenchmarkComparison from "./pages/BenchmarkComparison";

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<ReportViewer />} />
          <Route path="/benchmarks" element={<BenchmarkComparison />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
