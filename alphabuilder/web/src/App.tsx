import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DesignSystem from './design-system/DesignSystem';
import { MainLayout } from './components/layout/MainLayout';
import { Dashboard } from './pages/Dashboard';
import { Workspace } from './pages/Workspace';
import { DataLake } from './pages/DataLake';
import { NeuralNet } from './pages/NeuralNet';
import { Settings } from './pages/Settings';

function App() {
  return (
    <Router>
      <Routes>
        {/* Design System Route (Standalone) */}
        <Route path="/design" element={<DesignSystem />} />

        {/* Main Application Routes (Wrapped in Layout) */}
        <Route path="/" element={
          <MainLayout>
            <Dashboard />
          </MainLayout>
        } />

        <Route path="/workspace/:id" element={
          <MainLayout>
            <Workspace />
          </MainLayout>
        } />

        <Route path="/data" element={
          <MainLayout>
            <DataLake />
          </MainLayout>
        } />

        <Route path="/neural" element={
          <MainLayout>
            <NeuralNet />
          </MainLayout>
        } />

        <Route path="/settings" element={
          <MainLayout>
            <Settings />
          </MainLayout>
        } />

        {/* Fallback for dev convenience */}
        <Route path="/workspace" element={
          <MainLayout>
            <div className="flex items-center justify-center h-full text-white/40">
              Selecione um projeto no Painel
            </div>
          </MainLayout>
        } />
      </Routes>
    </Router>
  );
}

export default App;
