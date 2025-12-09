import { useLocation, useParams } from 'react-router-dom';
import { Bell, Search, Command, Home, Database, Box, Cpu, Settings, Play, ChevronRight } from 'lucide-react';

export const Header = ({ collapsed }: { collapsed: boolean }) => {
    const location = useLocation();
    const params = useParams<{ id?: string; dbId?: string; episodeId?: string }>();

    // Build breadcrumbs based on current route
    const getBreadcrumbs = () => {
        const path = location.pathname;
        const crumbs: { label: string; icon?: typeof Home; isActive?: boolean }[] = [];

        // Always start with "AlphaBuilder"
        crumbs.push({ label: 'AlphaBuilder', icon: Home });

        if (path === '/') {
            crumbs.push({ label: 'Painel', isActive: true });
        } else if (path.startsWith('/data')) {
            crumbs.push({ label: 'Dados de Treino', icon: Database });

            if (params.dbId) {
                crumbs.push({ label: params.dbId, isActive: !params.episodeId });

                if (params.episodeId) {
                    crumbs.push({
                        label: `Replay`,
                        icon: Play,
                        isActive: true,
                    });
                }
            } else {
                crumbs[crumbs.length - 1].isActive = true;
            }
        } else if (path.startsWith('/workspace')) {
            crumbs.push({ label: 'Laboratório', icon: Box });
            if (params.id) {
                crumbs.push({ label: params.id.slice(0, 12) + '...', isActive: true });
            }
        } else if (path === '/neural') {
            crumbs.push({ label: 'Rede Neural', icon: Cpu, isActive: true });
        } else if (path === '/settings') {
            crumbs.push({ label: 'Configurações', icon: Settings, isActive: true });
        }

        return crumbs;
    };

    const breadcrumbs = getBreadcrumbs();

    return (
        <header
            className="h-16 border-b border-white/5 bg-void/50 backdrop-blur-sm flex items-center justify-between px-8 fixed top-0 right-0 z-40 transition-all duration-300"
            style={{ left: collapsed ? 80 : 260 }}
        >
            {/* Breadcrumbs / Context */}
            <div className="flex items-center gap-1">
                {breadcrumbs.map((crumb, index) => (
                    <div key={index} className="flex items-center">
                        {index > 0 && (
                            <ChevronRight size={14} className="mx-2 text-white/20" />
                        )}
                        <span
                            className={`text-sm font-mono flex items-center gap-1.5 ${crumb.isActive ? 'text-white' : 'text-white/40'
                                }`}
                        >
                            {crumb.icon && <crumb.icon size={14} className={crumb.isActive ? 'text-cyan' : ''} />}
                            {crumb.label}
                        </span>
                    </div>
                ))}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-6">
                {/* Search Bar */}
                <div className="relative hidden md:block group">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 group-focus-within:text-cyan transition-colors" size={16} />
                    <input
                        type="text"
                        placeholder="Buscar..."
                        className="bg-white/5 border border-white/10 rounded-full py-1.5 pl-10 pr-12 text-sm text-white focus:outline-none focus:border-cyan/50 focus:bg-white/10 transition-all w-64"
                    />
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 text-white/20 text-xs font-mono border border-white/10 px-1.5 rounded">
                        <Command size={10} /> K
                    </div>
                </div>

                {/* Notifications */}
                <button className="relative text-white/60 hover:text-white transition-colors">
                    <Bell size={20} />
                    <span className="absolute -top-1 -right-1 w-2 h-2 bg-magenta rounded-full animate-pulse" />
                </button>

                {/* Status Indicator */}
                <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20">
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                    <span className="text-xs font-mono text-green-500 font-medium">MOTOR: ONLINE</span>
                </div>
            </div>
        </header>
    );
};
