import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    Gamepad2,
    Brain,
    Cpu,
    Clock,
    Trophy,
    Filter,
    Search,
    ChevronRight,
    AlertCircle,
    RefreshCw,
} from 'lucide-react';
import { fetchGames } from '../api/selfPlayService';
import type { GameSummary } from '../api/selfPlayService';

// =============================================================================
// Components
// =============================================================================

const FilterBar = ({
    engineFilter,
    setEngineFilter,
    versionFilter,
    setVersionFilter,
    search,
    setSearch,
}: {
    engineFilter: string;
    setEngineFilter: (v: string) => void;
    versionFilter: string;
    setVersionFilter: (v: string) => void;
    search: string;
    setSearch: (v: string) => void;
}) => (
    <div className="flex flex-wrap gap-4 mb-6">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30" />
            <input
                type="text"
                placeholder="Buscar por ID..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-white/30 focus:border-cyan/50 focus:outline-none"
            />
        </div>

        {/* Engine Filter */}
        <div className="flex items-center gap-2">
            <Cpu size={16} className="text-white/40" />
            <select
                value={engineFilter}
                onChange={(e) => setEngineFilter(e.target.value)}
                className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan/50 focus:outline-none"
            >
                <option value="all">Todos os Motores</option>
                <option value="simple">Simple Backbone</option>
                <option value="swin-unetr">Swin-UNETR</option>
            </select>
        </div>

        {/* Version Filter */}
        <div className="flex items-center gap-2">
            <Filter size={16} className="text-white/40" />
            <select
                value={versionFilter}
                onChange={(e) => setVersionFilter(e.target.value)}
                className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan/50 focus:outline-none"
            >
                <option value="all">Todas as Versões</option>
                <option value="warmup">Pós-Warmup</option>
                <option value="selfplay-v1">Self-Play v1</option>
                <option value="selfplay-v2">Self-Play v2</option>
            </select>
        </div>
    </div>
);

const GameCard = ({ game, deprecated }: { game: GameSummary; deprecated?: boolean }) => {
    const engineLabel = game.neural_engine === 'swin-unetr' ? 'Swin-UNETR' : 'Simple';
    const versionLabel = game.checkpoint_version.replace('selfplay-', 'SP-');
    const date = new Date(game.created_at);
    const score = game.final_score ?? 0;
    const volume = game.final_volume ?? 0;

    return (
        <Link to={`/games/${game.game_id}${deprecated ? '?deprecated=true' : ''}`}>
            <motion.div
                whileHover={{ scale: 1.02, y: -2 }}
                className="bg-matter/80 border border-white/10 rounded-xl p-5 hover:border-cyan/30 transition-colors group"
            >
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                    <div>
                        <div className="flex items-center gap-2 mb-1">
                            <Gamepad2 size={16} className="text-cyan" />
                            <span className="font-mono text-xs text-white/60">
                                {game.game_id.slice(-12)}
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className={`
                                text-xs px-2 py-0.5 rounded-full
                                ${game.neural_engine === 'swin-unetr'
                                    ? 'bg-purple/20 text-purple border border-purple/30'
                                    : 'bg-cyan/20 text-cyan border border-cyan/30'
                                }
                            `}>
                                {engineLabel}
                            </span>
                            <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-white/60">
                                {versionLabel}
                            </span>
                        </div>
                    </div>
                    <ChevronRight
                        size={20}
                        className="text-white/20 group-hover:text-cyan transition-colors"
                    />
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="bg-black/30 rounded-lg p-3">
                        <div className="flex items-center gap-1 text-white/40 text-xs mb-1">
                            <Trophy size={12} />
                            Score Final
                        </div>
                        <div className={`font-mono text-lg ${score >= 0 ? 'text-cyan' : 'text-orange-400'}`}>
                            {score.toFixed(3)}
                        </div>
                    </div>
                    <div className="bg-black/30 rounded-lg p-3">
                        <div className="flex items-center gap-1 text-white/40 text-xs mb-1">
                            <Brain size={12} />
                            Steps
                        </div>
                        <div className="font-mono text-lg text-white">
                            {game.total_steps}
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between text-xs text-white/40">
                    <div className="flex items-center gap-1">
                        <Clock size={12} />
                        {date.toLocaleDateString('pt-BR')} {date.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                    <div>
                        Vol: {(volume * 100).toFixed(1)}%
                    </div>
                </div>
            </motion.div>
        </Link>
    );
};

// =============================================================================
// Main Component
// =============================================================================

export const GamesList = () => {
    const [games, setGames] = useState<GameSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [engineFilter, setEngineFilter] = useState('all');
    const [versionFilter, setVersionFilter] = useState('all');
    const [search, setSearch] = useState('');
    const [useDeprecated, setUseDeprecated] = useState(false);

    const loadGames = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchGames(
                engineFilter !== 'all' ? engineFilter : undefined,
                versionFilter !== 'all' ? versionFilter : undefined,
                50,
                useDeprecated
            );
            setGames(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Erro ao carregar partidas');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadGames();
    }, [engineFilter, versionFilter, useDeprecated]);

    // Filter games
    const filteredGames = games.filter((g) => {
        if (search && !g.game_id.toLowerCase().includes(search.toLowerCase())) {
            return false;
        }
        if (engineFilter !== 'all' && g.neural_engine !== engineFilter) {
            return false;
        }
        if (versionFilter !== 'all' && g.checkpoint_version !== versionFilter) {
            return false;
        }
        return true;
    });

    return (
        <div className="min-h-[calc(100vh-64px)] p-8 bg-void">
            {/* Header with Deprecated Toggle */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8 flex justify-between items-start"
            >
                <div>
                    <h1 className="text-3xl font-display font-bold text-white mb-2">
                        Replay de <span className="text-cyan">Partidas</span>
                    </h1>
                    <p className="text-white/40">
                        Visualize e analise partidas de self-play com detalhes da busca MCTS
                    </p>
                </div>
                {/* Deprecated DB Toggle - Top Right */}
                <button
                    onClick={() => setUseDeprecated(!useDeprecated)}
                    className={`
                        px-4 py-2 rounded-lg font-mono text-sm transition-all flex items-center gap-2
                        ${useDeprecated
                            ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                            : 'bg-white/5 text-white/40 border border-white/10 hover:border-white/20'
                        }
                    `}
                >
                    <span className={`w-2 h-2 rounded-full ${useDeprecated ? 'bg-orange-400' : 'bg-white/30'}`} />
                    {useDeprecated ? '📦 Banco Deprecado (v1)' : '📁 Banco Atual'}
                </button>
            </motion.div>

            {/* Stats Summary */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="grid grid-cols-4 gap-4 mb-8"
            >
                <div className="bg-matter/60 border border-white/10 rounded-xl p-4">
                    <div className="text-white/40 text-sm mb-1">Total de Partidas</div>
                    <div className="text-2xl font-bold text-white">{games.length}</div>
                </div>
                <div className="bg-matter/60 border border-white/10 rounded-xl p-4">
                    <div className="text-white/40 text-sm mb-1">Melhor Score</div>
                    <div className="text-2xl font-bold text-cyan">
                        {games.length > 0 ? Math.max(...games.map(g => g.final_score ?? -999)).toFixed(3) : '-'}
                    </div>
                </div>
                <div className="bg-matter/60 border border-white/10 rounded-xl p-4">
                    <div className="text-white/40 text-sm mb-1">Swin-UNETR</div>
                    <div className="text-2xl font-bold text-purple">
                        {games.filter(g => g.neural_engine === 'swin-unetr').length}
                    </div>
                </div>
                <div className="bg-matter/60 border border-white/10 rounded-xl p-4">
                    <div className="text-white/40 text-sm mb-1">Simple Backbone</div>
                    <div className="text-2xl font-bold text-green-400">
                        {games.filter(g => g.neural_engine === 'simple').length}
                    </div>
                </div>
            </motion.div>

            {/* Filters */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
                className="flex flex-col gap-4"
            >
                <FilterBar
                    engineFilter={engineFilter}
                    setEngineFilter={setEngineFilter}
                    versionFilter={versionFilter}
                    setVersionFilter={setVersionFilter}
                    search={search}
                    setSearch={setSearch}
                />

            </motion.div>

            {/* Games Grid */}
            {loading ? (
                <div className="flex items-center justify-center h-64">
                    <div className="w-8 h-8 border-2 border-cyan/30 border-t-cyan rounded-full animate-spin" />
                </div>
            ) : error ? (
                <div className="text-center py-16">
                    <AlertCircle size={48} className="mx-auto mb-4 text-red-400 opacity-60" />
                    <p className="text-white/60 mb-4">{error}</p>
                    <button
                        onClick={loadGames}
                        className="flex items-center gap-2 mx-auto px-4 py-2 bg-cyan/20 text-cyan rounded-lg hover:bg-cyan/30 transition-colors"
                    >
                        <RefreshCw size={16} />
                        Tentar novamente
                    </button>
                </div>
            ) : filteredGames.length === 0 ? (
                <div className="text-center py-16 text-white/40">
                    <Gamepad2 size={48} className="mx-auto mb-4 opacity-30" />
                    <p>Nenhuma partida encontrada</p>
                </div>
            ) : (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
                >
                    {filteredGames.map((game, i) => (
                        <motion.div
                            key={game.game_id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.05 * i }}
                        >
                            <GameCard game={game} deprecated={useDeprecated} />
                        </motion.div>
                    ))}
                </motion.div>
            )}
        </div>
    );
};
