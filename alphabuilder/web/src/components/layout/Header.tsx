import { Bell, Search, Command } from 'lucide-react';

export const Header = () => {
    return (
        <header className="h-16 border-b border-white/5 bg-void/50 backdrop-blur-sm flex items-center justify-between px-8 fixed top-0 right-0 left-0 z-40 ml-[80px] lg:ml-[260px] transition-all">
            {/* Breadcrumbs / Context */}
            <div className="flex items-center gap-4">
                <div className="text-white/40 text-sm font-mono">
                    Início <span className="mx-2">/</span> Projeto Alpha <span className="mx-2">/</span> <span className="text-white">Estudo #001</span>
                </div>
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
