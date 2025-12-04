import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link, useLocation } from 'react-router-dom';
import {
    LayoutDashboard,
    Box,
    Database,
    Cpu,
    Settings,
    ChevronLeft,
    ChevronRight,
    Hexagon
} from 'lucide-react';

const NavItem = ({ icon: Icon, label, path, collapsed, active }: { icon: any, label: string, path: string, collapsed: boolean, active: boolean }) => (
    <Link to={path}>
        <motion.div
            className={`flex items-center gap-4 p-3 rounded-lg mb-2 transition-colors relative group ${active ? 'bg-cyan/10 text-cyan' : 'text-white/60 hover:text-white hover:bg-white/5'}`}
            whileHover={{ x: 4 }}
        >
            <Icon size={20} strokeWidth={1.5} />

            {!collapsed && (
                <motion.span
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="text-sm font-medium whitespace-nowrap"
                >
                    {label}
                </motion.span>
            )}

            {/* Active Indicator */}
            {active && (
                <motion.div
                    layoutId="activeNav"
                    className="absolute left-0 top-0 bottom-0 w-1 bg-cyan rounded-r-full"
                />
            )}

            {/* Tooltip for collapsed state */}
            {collapsed && (
                <div className="absolute left-full ml-4 px-2 py-1 bg-matter border border-white/10 rounded text-xs text-white opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50">
                    {label}
                </div>
            )}
        </motion.div>
    </Link>
);

export const Sidebar = ({ collapsed, setCollapsed }: { collapsed: boolean; setCollapsed: (v: boolean) => void }) => {
    const location = useLocation();

    return (
        <motion.aside
            animate={{ width: collapsed ? 80 : 260 }}
            className="h-screen bg-matter/80 backdrop-blur-xl border-r border-white/5 flex flex-col relative z-50 shrink-0"
        >
            {/* Logo Area */}
            <div className="h-20 flex items-center justify-center border-b border-white/5 relative">
                <div className="flex items-center gap-3 overflow-hidden px-6 w-full">
                    <div className="text-cyan shrink-0">
                        <Hexagon size={32} strokeWidth={1.5} className="animate-pulse-slow" />
                    </div>
                    {!collapsed && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="font-display font-bold text-xl text-white tracking-tight whitespace-nowrap"
                        >
                            Alpha<span className="text-cyan">Builder</span>
                        </motion.div>
                    )}
                </div>

                {/* Collapse Toggle */}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    className="absolute -right-3 top-8 w-6 h-6 bg-steel border border-white/10 rounded-full flex items-center justify-center text-white/60 hover:text-white hover:bg-cyan hover:text-black transition-colors z-50"
                >
                    {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
                </button>
            </div>

            {/* Navigation */}
            <div className="flex-1 py-8 px-4 overflow-y-auto overflow-x-hidden">
                <div className="mb-8">
                    {!collapsed && <h3 className="text-xs font-mono text-white/30 mb-4 px-2 uppercase whitespace-nowrap">Principal</h3>}
                    <NavItem icon={LayoutDashboard} label="Painel" path="/" collapsed={collapsed} active={location.pathname === '/'} />
                    <NavItem icon={Box} label="Laboratório" path="/workspace" collapsed={collapsed} active={location.pathname.includes('/workspace')} />
                    <NavItem icon={Database} label="Dados de Treino" path="/data" collapsed={collapsed} active={location.pathname.startsWith('/data')} />
                </div>

                <div>
                    {!collapsed && <h3 className="text-xs font-mono text-white/30 mb-4 px-2 uppercase whitespace-nowrap">Sistema</h3>}
                    <NavItem icon={Cpu} label="Rede Neural" path="/neural" collapsed={collapsed} active={location.pathname === '/neural'} />
                    <NavItem icon={Settings} label="Configurações" path="/settings" collapsed={collapsed} active={location.pathname === '/settings'} />
                </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-white/5">
                <div className={`flex items-center gap-3 ${collapsed ? 'justify-center' : ''}`}>
                    <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan to-purple border border-white/20 shrink-0" />
                    {!collapsed && (
                        <div className="overflow-hidden">
                            <p className="text-sm text-white font-medium truncate">Gustavo Mello</p>
                            <p className="text-xs text-green-400 flex items-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                                Online
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </motion.aside>
    );
};
