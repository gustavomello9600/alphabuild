import { useEffect, useState, useMemo, useRef } from 'react';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Database,
    ChevronRight,
    ChevronLeft,
    Play,
    Layers,
    Trophy,
    ArrowLeft,
    RefreshCw,
    HardDrive,
    Hash,
    AlertCircle,
    Network,
    Sparkles,
    Search,
    ChevronsLeft,
    ChevronsRight,
} from 'lucide-react';
import {
    fetchDatabases,
    fetchEpisodes,
    type DatabaseInfo,
    type EpisodeSummary,
} from '../api/trainingDataService';

// --- Constants ---
const EPISODES_PER_PAGE = 50;

// --- Animation Variants ---
const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.03,
        },
    },
};

const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 },
};

// --- Database Card Component ---
const DatabaseCard = ({
    db,
    onClick,
    isSelected,
}: {
    db: DatabaseInfo;
    onClick: () => void;
    isSelected: boolean;
}) => (
    <motion.div
        variants={itemVariants}
        whileHover={{ scale: 1.02, y: -4 }}
        whileTap={{ scale: 0.98 }}
        onClick={onClick}
        className={`
            relative p-6 rounded-xl cursor-pointer transition-all duration-300
            border backdrop-blur-sm group
            ${isSelected
                ? 'bg-cyan/10 border-cyan/40 shadow-[0_0_30px_rgba(0,240,255,0.15)]'
                : 'bg-matter/40 border-white/5 hover:border-cyan/30 hover:bg-matter/60'
            }
        `}
    >
        <div
            className={`
                p-3 rounded-xl mb-4 inline-flex transition-colors
                ${isSelected
                    ? 'bg-cyan/20 text-cyan'
                    : 'bg-white/5 text-white/60 group-hover:bg-cyan/10 group-hover:text-cyan'
                }
            `}
        >
            <Database size={24} strokeWidth={1.5} />
        </div>

        <h3
            className={`
                font-display font-bold text-lg mb-2 transition-colors
                ${isSelected ? 'text-cyan' : 'text-white group-hover:text-cyan'}
            `}
        >
            {db.name}
        </h3>

        <p className="text-xs font-mono text-white/30 mb-4 truncate" title={db.path}>
            {db.path}
        </p>

        <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5 text-white/50">
                <Layers size={14} />
                <span>{db.episode_count.toLocaleString()} episódios</span>
            </div>
            <div className="flex items-center gap-1.5 text-white/50">
                <HardDrive size={14} />
                <span>{db.size_mb} MB</span>
            </div>
        </div>

        {isSelected && (
            <motion.div
                layoutId="selectedDb"
                className="absolute inset-0 rounded-xl border-2 border-cyan pointer-events-none"
            />
        )}

        <ChevronRight
            size={20}
            className={`
                absolute right-4 top-1/2 -translate-y-1/2 transition-all
                ${isSelected
                    ? 'text-cyan opacity-100 translate-x-0'
                    : 'text-white/20 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0'
                }
            `}
        />
    </motion.div>
);

// --- Episode Row Component ---
const EpisodeRow = ({
    episode,
    onClick,
}: {
    episode: EpisodeSummary;
    onClick: () => void;
}) => {
    const hasGoodCompliance = episode.final_compliance !== null && episode.final_compliance < 100;
    const qualityColor = hasGoodCompliance ? 'text-green-400' : 'text-white/40';

    return (
        <motion.tr
            variants={itemVariants}
            onClick={onClick}
            className="group cursor-pointer hover:bg-white/5 transition-colors border-b border-white/5 last:border-0"
        >
            <td className="p-3">
                <div className="flex items-center gap-3">
                    <div className="p-1.5 rounded-lg bg-purple/10 text-purple group-hover:bg-cyan/10 group-hover:text-cyan transition-colors">
                        <Hash size={14} />
                    </div>
                    <div>
                        <div className="font-mono text-sm text-white group-hover:text-cyan transition-colors truncate max-w-[180px]">
                            {episode.episode_id.slice(0, 12)}...
                        </div>
                        <div className="text-[10px] text-white/30 font-mono">
                            {episode.total_steps} passos
                        </div>
                    </div>
                </div>
            </td>

            <td className="p-3">
                <div className="flex items-center gap-2">
                    <Network size={12} className="text-green-400" />
                    <span className="font-mono text-sm text-white/70">{episode.steps_phase1}</span>
                </div>
            </td>

            <td className="p-3">
                <div className="flex items-center gap-2">
                    <Sparkles size={12} className="text-purple" />
                    <span className="font-mono text-sm text-white/70">{episode.steps_phase2}</span>
                </div>
            </td>

            <td className="p-3">
                <span className={`font-mono text-sm ${qualityColor}`}>
                    {episode.final_compliance !== null
                        ? episode.final_compliance.toFixed(1)
                        : '—'
                    }
                </span>
            </td>

            <td className="p-3">
                <span className="font-mono text-sm text-white/70">
                    {episode.final_volume_fraction !== null
                        ? `${(episode.final_volume_fraction * 100).toFixed(1)}%`
                        : '—'
                    }
                </span>
            </td>

            <td className="p-3">
                <span className={`font-mono text-sm ${episode.final_reward && episode.final_reward > 0 ? 'text-green-400' : 'text-white/50'}`}>
                    {episode.final_reward !== null
                        ? (episode.final_reward > 0 ? '+' : '') + episode.final_reward.toFixed(3)
                        : '—'
                    }
                </span>
            </td>

            <td className="p-3 text-right">
                <motion.button
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    className="p-2 rounded-lg bg-cyan/10 text-cyan opacity-0 group-hover:opacity-100 transition-opacity"
                >
                    <Play size={14} fill="currentColor" />
                </motion.button>
            </td>
        </motion.tr>
    );
};

// --- Pagination Component ---
const Pagination = ({
    currentPage,
    totalPages,
    onPageChange,
}: {
    currentPage: number;
    totalPages: number;
    onPageChange: (page: number) => void;
}) => {
    const pageNumbers = useMemo(() => {
        const pages: (number | string)[] = [];
        const showPages = 5;
        
        if (totalPages <= showPages + 2) {
            for (let i = 1; i <= totalPages; i++) pages.push(i);
        } else {
            pages.push(1);
            
            if (currentPage > 3) pages.push('...');
            
            const start = Math.max(2, currentPage - 1);
            const end = Math.min(totalPages - 1, currentPage + 1);
            
            for (let i = start; i <= end; i++) pages.push(i);
            
            if (currentPage < totalPages - 2) pages.push('...');
            
            pages.push(totalPages);
        }
        
        return pages;
    }, [currentPage, totalPages]);

    if (totalPages <= 1) return null;

    return (
        <div className="flex items-center justify-center gap-2 mt-6">
            <button
                onClick={() => onPageChange(1)}
                disabled={currentPage === 1}
                className="p-2 rounded-lg bg-white/5 text-white/40 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
                <ChevronsLeft size={16} />
            </button>
            <button
                onClick={() => onPageChange(currentPage - 1)}
                disabled={currentPage === 1}
                className="p-2 rounded-lg bg-white/5 text-white/40 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
                <ChevronLeft size={16} />
            </button>

            <div className="flex items-center gap-1">
                {pageNumbers.map((page, idx) => (
                    page === '...' ? (
                        <span key={`dots-${idx}`} className="px-2 text-white/30">...</span>
                    ) : (
                        <button
                            key={page}
                            onClick={() => onPageChange(page as number)}
                            className={`
                                min-w-[36px] h-9 rounded-lg font-mono text-sm transition-colors
                                ${currentPage === page
                                    ? 'bg-cyan text-black font-bold'
                                    : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white'
                                }
                            `}
                        >
                            {page}
                        </button>
                    )
                ))}
            </div>

            <button
                onClick={() => onPageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="p-2 rounded-lg bg-white/5 text-white/40 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
                <ChevronRight size={16} />
            </button>
            <button
                onClick={() => onPageChange(totalPages)}
                disabled={currentPage === totalPages}
                className="p-2 rounded-lg bg-white/5 text-white/40 hover:bg-white/10 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
                <ChevronsRight size={16} />
            </button>
        </div>
    );
};

// --- Loading Skeleton ---
const LoadingSkeleton = () => (
    <div className="space-y-4">
        {[1, 2, 3].map((i) => (
            <div key={i} className="h-24 bg-white/5 rounded-xl animate-pulse" />
        ))}
    </div>
);

// --- Empty State ---
const EmptyState = ({ message, icon: Icon }: { message: string; icon: typeof Database }) => (
    <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="flex flex-col items-center justify-center py-20 text-center"
    >
        <div className="p-6 rounded-full bg-white/5 text-white/20 mb-6">
            <Icon size={48} strokeWidth={1} />
        </div>
        <p className="text-white/40 text-lg">{message}</p>
    </motion.div>
);

// --- Main Component ---
export const TrainingData = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const { dbId } = useParams<{ dbId?: string }>();

    const [databases, setDatabases] = useState<DatabaseInfo[]>([]);
    const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
    const [loadingDbs, setLoadingDbs] = useState(true);
    const [loadingEpisodes, setLoadingEpisodes] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    // Pagination state
    const [currentPage, setCurrentPage] = useState(1);
    const [searchQuery, setSearchQuery] = useState('');
    
    // Track previous location to detect back navigation
    const previousPathRef = useRef<string>('');
    const loadedDbIdRef = useRef<string | null>(null);

    // Filter and paginate episodes
    const filteredEpisodes = useMemo(() => {
        if (!searchQuery.trim()) return episodes;
        const query = searchQuery.toLowerCase();
        return episodes.filter(ep => 
            ep.episode_id.toLowerCase().includes(query)
        );
    }, [episodes, searchQuery]);

    const totalPages = Math.ceil(filteredEpisodes.length / EPISODES_PER_PAGE);
    
    const paginatedEpisodes = useMemo(() => {
        const start = (currentPage - 1) * EPISODES_PER_PAGE;
        return filteredEpisodes.slice(start, start + EPISODES_PER_PAGE);
    }, [filteredEpisodes, currentPage]);

    // Reset page when search changes
    useEffect(() => {
        setCurrentPage(1);
    }, [searchQuery]);

    // Load databases on mount
    useEffect(() => {
        loadDatabases();
    }, []);

    // Detect navigation changes and force reload when coming back from replay
    useEffect(() => {
        const currentPath = location.pathname;
        const previousPath = previousPathRef.current;
        
        // Check if we're navigating back from a replay page to episode list
        const wasOnReplay = previousPath.includes('/episode/');
        const isOnEpisodeList = currentPath.match(/^\/data\/[^/]+$/);
        
        console.log(`[TrainingData] Navigation: ${previousPath} -> ${currentPath}`);
        console.log(`[TrainingData] Was on replay: ${wasOnReplay}, Is on episode list: ${isOnEpisodeList}, dbId: ${dbId}`);
        
        // If we came back from replay to episode list, force reload
        if (wasOnReplay && isOnEpisodeList && dbId) {
            console.log(`[TrainingData] Detected back navigation from replay, forcing reload for: ${dbId}`);
            // Reset the loaded ref to force reload
            loadedDbIdRef.current = null;
            // Clear episodes first to trigger UI update
            setEpisodes([]);
            // Force reload episodes
            setTimeout(() => {
                loadEpisodes(dbId);
            }, 50);
        }
        
        previousPathRef.current = currentPath;
    }, [location.pathname, dbId]);

    // Load episodes when dbId changes (from URL) or on mount
    useEffect(() => {
        console.log(`[TrainingData] dbId effect: ${dbId}, location: ${location.pathname}, loadedDbId: ${loadedDbIdRef.current}`);
        
        if (dbId) {
            // Always reload if dbId changed or if we haven't loaded it yet
            if (loadedDbIdRef.current !== dbId) {
                console.log(`[TrainingData] Loading episodes for dbId: ${dbId}`);
                loadedDbIdRef.current = dbId;
                loadEpisodes(dbId);
            } else {
                console.log(`[TrainingData] Already loaded episodes for dbId: ${dbId}`);
                // Even if already loaded, check if episodes array is empty (might have been cleared)
                if (episodes.length === 0 && !loadingEpisodes) {
                    console.log(`[TrainingData] Episodes array is empty, reloading...`);
                    loadEpisodes(dbId);
                }
            }
        } else {
            setEpisodes([]);
            setCurrentPage(1);
            setSearchQuery('');
            loadedDbIdRef.current = null;
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dbId]);

    const loadDatabases = async () => {
        setLoadingDbs(true);
        setError(null);
        try {
            const data = await fetchDatabases();
            setDatabases(data);
        } catch (err) {
            setError('Não foi possível conectar ao backend. Certifique-se de que o servidor está rodando.');
            console.error('Failed to fetch databases:', err);
        } finally {
            setLoadingDbs(false);
        }
    };

    const loadEpisodes = async (databaseId: string) => {
        setLoadingEpisodes(true);
        setError(null);
        setCurrentPage(1);
        setSearchQuery('');
        try {
            console.log(`[TrainingData] Loading episodes for database: ${databaseId}`);
            const data = await fetchEpisodes(databaseId);
            console.log(`[TrainingData] Loaded ${data.length} episodes`);
            setEpisodes(data);
        } catch (err) {
            console.error('[TrainingData] Failed to fetch episodes:', err);
            const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
            setError(`Erro ao carregar episódios: ${errorMessage}. Verifique se o backend está rodando.`);
            setEpisodes([]);
        } finally {
            setLoadingEpisodes(false);
        }
    };

    const handleSelectDb = (databaseId: string) => {
        // Reset state when selecting a new database
        loadedDbIdRef.current = null;
        navigate(`/data/${databaseId}`);
    };

    const handleBackToList = () => {
        navigate('/data');
    };

    const handlePlayEpisode = (episodeId: string) => {
        if (dbId) {
            navigate(`/data/${dbId}/episode/${episodeId}`);
        }
    };

    const handlePageChange = (page: number) => {
        setCurrentPage(page);
        // Scroll to top of table
        window.scrollTo({ top: 200, behavior: 'smooth' });
    };

    // Safety check: if we're on episode list route but episodes are empty, reload
    useEffect(() => {
        const isOnEpisodeList = location.pathname.match(/^\/data\/[^/]+$/);
        if (isOnEpisodeList && dbId && episodes.length === 0 && !loadingEpisodes && !loadingDbs) {
            console.log(`[TrainingData] Safety check: episodes empty on episode list route, reloading for: ${dbId}`);
            loadedDbIdRef.current = null;
            loadEpisodes(dbId);
        }
    }, [location.pathname, dbId, episodes.length, loadingEpisodes, loadingDbs]);

    // Stats calculations
    const stats = useMemo(() => {
        if (episodes.length === 0) return null;
        
        const avgReward = episodes.reduce((sum, e) => sum + (e.final_reward || 0), 0) / episodes.length;
        const totalPhase1 = episodes.reduce((sum, e) => sum + e.steps_phase1, 0);
        const totalPhase2 = episodes.reduce((sum, e) => sum + e.steps_phase2, 0);
        
        return {
            total: episodes.length,
            avgReward,
            totalPhase1,
            totalPhase2,
        };
    }, [episodes]);

    return (
        <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="flex justify-between items-end mb-12">
                <div className="flex items-center gap-4">
                    {dbId && (
                        <motion.button
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            onClick={handleBackToList}
                            className="p-3 rounded-xl bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors"
                        >
                            <ArrowLeft size={20} />
                        </motion.button>
                    )}
                    <div>
                        <h1 className="text-4xl font-display font-bold text-white mb-2">
                            Dados de{' '}
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan to-purple">
                                Treino
                            </span>
                        </h1>
                        <p className="text-white/60 max-w-xl">
                            {dbId
                                ? `Visualize e reproduza episódios de treinamento do database ${dbId}.`
                                : 'Explore os databases de treinamento disponíveis no projeto.'}
                        </p>
                    </div>
                </div>

                <button
                    onClick={() => dbId ? loadEpisodes(dbId) : loadDatabases()}
                    disabled={loadingDbs || loadingEpisodes}
                    className="bg-white/5 hover:bg-white/10 disabled:opacity-50 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
                >
                    <RefreshCw size={16} className={(loadingDbs || loadingEpisodes) ? 'animate-spin' : ''} />
                    Atualizar
                </button>
            </div>

            {/* Error State */}
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="mb-8 p-4 rounded-xl bg-magenta/10 border border-magenta/30 flex items-center gap-3"
                    >
                        <AlertCircle className="text-magenta shrink-0" size={20} />
                        <p className="text-magenta/80 text-sm">{error}</p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Content */}
            <AnimatePresence mode="wait">
                {!dbId ? (
                    // Database Selection View
                    <motion.div
                        key="databases"
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        variants={containerVariants}
                    >
                        {loadingDbs ? (
                            <LoadingSkeleton />
                        ) : databases.length === 0 ? (
                            <EmptyState
                                icon={Database}
                                message="Nenhum database encontrado. Verifique se existem arquivos .db na raiz do projeto ou na pasta data."
                            />
                        ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                {databases.map((db) => (
                                    <DatabaseCard
                                        key={db.id}
                                        db={db}
                                        onClick={() => handleSelectDb(db.id)}
                                        isSelected={false}
                                    />
                                ))}
                            </div>
                        )}
                    </motion.div>
                ) : (
                    // Episode List View
                    <motion.div
                        key="episodes"
                        initial="hidden"
                        animate="visible"
                        exit="hidden"
                        variants={containerVariants}
                    >
                        {/* Stats Bar */}
                        {stats && (
                            <motion.div
                                variants={itemVariants}
                                className="grid grid-cols-4 gap-4 mb-6"
                            >
                                <div className="p-4 rounded-xl bg-matter/40 border border-white/5">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Layers size={16} className="text-cyan" />
                                        <span className="text-xs text-white/40 uppercase tracking-wider">
                                            Total
                                        </span>
                                    </div>
                                    <div className="text-2xl font-display font-bold text-cyan">
                                        {stats.total.toLocaleString()}
                                    </div>
                                </div>
                                <div className="p-4 rounded-xl bg-matter/40 border border-white/5">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Network size={16} className="text-green-400" />
                                        <span className="text-xs text-white/40 uppercase tracking-wider">
                                            Passos Conexão
                                        </span>
                                    </div>
                                    <div className="text-2xl font-display font-bold text-green-400">
                                        {stats.totalPhase1.toLocaleString()}
                                    </div>
                                </div>
                                <div className="p-4 rounded-xl bg-matter/40 border border-white/5">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Sparkles size={16} className="text-purple" />
                                        <span className="text-xs text-white/40 uppercase tracking-wider">
                                            Passos Refinamento
                                        </span>
                                    </div>
                                    <div className="text-2xl font-display font-bold text-purple">
                                        {stats.totalPhase2.toLocaleString()}
                                    </div>
                                </div>
                                <div className="p-4 rounded-xl bg-matter/40 border border-white/5">
                                    <div className="flex items-center gap-2 mb-2">
                                        <Trophy size={16} className="text-cyan" />
                                        <span className="text-xs text-white/40 uppercase tracking-wider">
                                            Recompensa Final Média
                                        </span>
                                    </div>
                                    <div className={`text-2xl font-display font-bold ${stats.avgReward > 0 ? 'text-green-400' : 'text-white/60'}`}>
                                        {stats.avgReward > 0 ? '+' : ''}{stats.avgReward.toFixed(3)}
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Search Bar */}
                        <motion.div variants={itemVariants} className="mb-4">
                            <div className="relative">
                                <Search size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-white/30" />
                                <input
                                    type="text"
                                    placeholder="Buscar por ID do episódio..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full bg-matter/40 border border-white/10 rounded-xl py-3 pl-11 pr-4 text-white placeholder-white/30 focus:outline-none focus:border-cyan/50 transition-colors"
                                />
                                {searchQuery && (
                                    <span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-white/40">
                                        {filteredEpisodes.length} resultados
                                    </span>
                                )}
                            </div>
                        </motion.div>

                        {/* Episodes Table */}
                        {loadingEpisodes ? (
                            <LoadingSkeleton />
                        ) : filteredEpisodes.length === 0 ? (
                            <EmptyState
                                icon={Layers}
                                message={searchQuery ? "Nenhum episódio encontrado com este filtro." : "Nenhum episódio encontrado neste database."}
                            />
                        ) : (
                            <>
                                <motion.div
                                    variants={itemVariants}
                                    className="bg-matter/30 border border-white/5 rounded-xl overflow-hidden"
                                >
                                    <table className="w-full">
                                        <thead className="bg-white/5 text-xs uppercase font-mono text-white/40">
                                            <tr>
                                                <th className="p-3 text-left">Episode ID</th>
                                                <th className="p-3 text-left">
                                                    <div className="flex items-center gap-1">
                                                        <Network size={10} className="text-green-400" />
                                                        Conexão
                                                    </div>
                                                </th>
                                                <th className="p-3 text-left">
                                                    <div className="flex items-center gap-1">
                                                        <Sparkles size={10} className="text-purple" />
                                                        Refinamento
                                                    </div>
                                                </th>
                                                <th className="p-3 text-left">Compliance</th>
                                                <th className="p-3 text-left">Volume</th>
                                                <th className="p-3 text-left">Recompensa</th>
                                                <th className="p-3 text-right">Replay</th>
                                            </tr>
                                        </thead>
                                        <motion.tbody
                                            key={currentPage}
                                            initial="hidden"
                                            animate="visible"
                                            variants={containerVariants}
                                        >
                                            {paginatedEpisodes.map((episode) => (
                                                <EpisodeRow
                                                    key={episode.episode_id}
                                                    episode={episode}
                                                    onClick={() => handlePlayEpisode(episode.episode_id)}
                                                />
                                            ))}
                                        </motion.tbody>
                                    </table>
                                </motion.div>

                                {/* Pagination */}
                                <Pagination
                                    currentPage={currentPage}
                                    totalPages={totalPages}
                                    onPageChange={handlePageChange}
                                />

                                {/* Page Info */}
                                <div className="text-center mt-4 text-xs text-white/30 font-mono">
                                    Mostrando {((currentPage - 1) * EPISODES_PER_PAGE) + 1} - {Math.min(currentPage * EPISODES_PER_PAGE, filteredEpisodes.length)} de {filteredEpisodes.length} episódios
                                </div>
                            </>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
