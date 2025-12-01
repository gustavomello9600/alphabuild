import { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Download, RefreshCw } from 'lucide-react';

interface Dataset {
    id: string;
    name: string;
    size: string;
    episodes: number;
    status: 'READY' | 'PROCESSING' | 'ARCHIVED';
    last_updated: string;
}

// Mock Data based on real files found in project
const MOCK_DATASETS: Dataset[] = [
    { id: '1', name: 'training_data.db', size: '2.4 GB', episodes: 15420, status: 'READY', last_updated: '2023-10-27' },
    { id: '2', name: 'harvest_part1.db', size: '850 MB', episodes: 5200, status: 'READY', last_updated: '2023-10-25' },
    { id: '3', name: 'local_test_selfplay.db', size: '120 MB', episodes: 850, status: 'PROCESSING', last_updated: 'Hoje' },
    { id: '4', name: 'simp_verify.db', size: '45 MB', episodes: 120, status: 'ARCHIVED', last_updated: '2023-10-20' },
    { id: '5', name: 'test_mining_final.db', size: '320 MB', episodes: 2100, status: 'READY', last_updated: '2023-10-22' },
];

export const DataLake = () => {
    const [datasets] = useState(MOCK_DATASETS);

    return (
        <div className="max-w-7xl mx-auto">
            <div className="flex justify-between items-end mb-12">
                <div>
                    <h1 className="text-4xl font-display font-bold text-white mb-2">
                        Data <span className="text-cyan">Lake</span>
                    </h1>
                    <p className="text-white/60 max-w-xl">
                        Gerencie os conjuntos de dados de treinamento e validação.
                    </p>
                </div>
                <button className="bg-white/5 hover:bg-white/10 text-white px-4 py-2 rounded flex items-center gap-2 transition-colors">
                    <RefreshCw size={16} /> Atualizar
                </button>
            </div>

            <div className="bg-matter/30 border border-white/5 rounded-xl overflow-hidden">
                <table className="w-full text-left">
                    <thead className="bg-white/5 text-xs uppercase font-mono text-white/40">
                        <tr>
                            <th className="p-6">Nome do Arquivo</th>
                            <th className="p-6">Tamanho</th>
                            <th className="p-6">Episódios</th>
                            <th className="p-6">Status</th>
                            <th className="p-6 text-right">Ações</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                        {datasets.map((ds) => (
                            <motion.tr
                                key={ds.id}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className="hover:bg-white/5 transition-colors group"
                            >
                                <td className="p-6">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-cyan/10 rounded text-cyan">
                                            <Database size={20} />
                                        </div>
                                        <div>
                                            <div className="font-medium text-white group-hover:text-cyan transition-colors">{ds.name}</div>
                                            <div className="text-xs text-white/40">{ds.last_updated}</div>
                                        </div>
                                    </div>
                                </td>
                                <td className="p-6 font-mono text-white/60">{ds.size}</td>
                                <td className="p-6 font-mono text-white/60">{ds.episodes.toLocaleString()}</td>
                                <td className="p-6">
                                    <span className={`text-xs font-bold px-2 py-1 rounded border ${ds.status === 'READY' ? 'bg-green-500/10 border-green-500/20 text-green-400' :
                                        ds.status === 'PROCESSING' ? 'bg-cyan/10 border-cyan/20 text-cyan animate-pulse' :
                                            'bg-white/5 border-white/10 text-white/40'
                                        }`}>
                                        {ds.status === 'READY' ? 'PRONTO' : ds.status === 'PROCESSING' ? 'PROCESSANDO' : 'ARQUIVADO'}
                                    </span>
                                </td>
                                <td className="p-6 text-right">
                                    <button className="p-2 hover:bg-white/10 rounded text-white/40 hover:text-white transition-colors">
                                        <Download size={18} />
                                    </button>
                                </td>
                            </motion.tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
