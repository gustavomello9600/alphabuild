import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Plus, Clock, MoreVertical, Folder, ArrowRight } from 'lucide-react';
import { mockService } from '../api/mockService';
import type { Project } from '../api/types';
import { Link } from 'react-router-dom';

const ProjectCard = ({ project }: { project: Project }) => (
    <motion.div
        whileHover={{ y: -5 }}
        className="group bg-matter/30 border border-white/5 rounded-xl overflow-hidden hover:border-cyan/30 transition-all cursor-pointer relative"
    >
        {/* Thumbnail Placeholder */}
        <div className="h-40 bg-gradient-to-br from-black/40 to-black/10 relative flex items-center justify-center">
            <div className="w-16 h-16 border border-white/10 rounded flex items-center justify-center text-white/20 group-hover:text-cyan/50 transition-colors">
                <Folder size={32} />
            </div>
            {/* Status Badge */}
            <div className={`absolute top-3 right-3 px-2 py-1 rounded text-xs font-mono border ${project.status === 'COMPLETED' ? 'bg-green-500/10 border-green-500/20 text-green-400' :
                project.status === 'IN_PROGRESS' ? 'bg-cyan/10 border-cyan/20 text-cyan' :
                    'bg-red-500/10 border-red-500/20 text-red-400'
                }`}>
                {project.status}
            </div>
        </div>

        <div className="p-5">
            <div className="flex justify-between items-start mb-2">
                <h3 className="text-lg font-display font-bold text-white group-hover:text-cyan transition-colors">{project.name}</h3>
                <button className="text-white/20 hover:text-white transition-colors">
                    <MoreVertical size={16} />
                </button>
            </div>
            <p className="text-xs text-white/40 font-mono mb-4 flex items-center gap-1">
                <Clock size={12} /> Editado em: {project.last_modified}
            </p>

            <Link to={`/workspace/${project.episode_id}`} className="inline-flex items-center gap-2 text-sm text-white/60 hover:text-white transition-colors">
                Abrir Laboratório <ArrowRight size={14} />
            </Link>
        </div>
    </motion.div>
);

export const Dashboard = () => {
    const [projects, setProjects] = useState<Project[]>([]);

    useEffect(() => {
        mockService.getProjects().then(setProjects);
    }, []);

    return (
        <div className="max-w-7xl mx-auto">
            {/* Hero Section */}
            <div className="mb-12 flex justify-between items-end">
                <div>
                    <h1 className="text-4xl font-display font-bold text-white mb-2">
                        Bem-vindo, <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan to-purple">Engenheiro</span>.
                    </h1>
                    <p className="text-white/60 max-w-xl">
                        Selecione um projeto para continuar a otimização ou inicie um novo estudo topológico.
                    </p>
                </div>

                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="bg-cyan text-black font-bold py-3 px-6 rounded-lg flex items-center gap-2 shadow-[0_0_20px_rgba(0,240,255,0.3)] hover:shadow-[0_0_30px_rgba(0,240,255,0.5)] transition-shadow"
                >
                    <Plus size={20} /> Novo Projeto
                </motion.button>
            </div>

            {/* Projects Grid */}
            <h2 className="text-sm font-mono text-white/40 uppercase mb-6 tracking-wider">Projetos Recentes</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {projects.map(p => (
                    <ProjectCard key={p.id} project={p} />
                ))}

                {/* New Project Placeholder Card */}
                <motion.div
                    whileHover={{ scale: 1.02 }}
                    className="border-2 border-dashed border-white/10 rounded-xl flex flex-col items-center justify-center p-8 cursor-pointer hover:border-cyan/30 hover:bg-cyan/5 transition-all group"
                >
                    <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4 group-hover:bg-cyan/20 transition-colors">
                        <Plus size={32} className="text-white/20 group-hover:text-cyan" />
                    </div>
                    <p className="text-white/40 font-medium group-hover:text-white">Criar Novo Estudo</p>
                </motion.div>
            </div>
        </div>
    );
};
