import { Save, Monitor, Globe, Server } from 'lucide-react';

export const Settings = () => {
    return (
        <div className="max-w-3xl mx-auto">
            <div className="mb-12">
                <h1 className="text-4xl font-display font-bold text-white mb-2">
                    Configurações
                </h1>
                <p className="text-white/60">
                    Preferências do sistema e conexões.
                </p>
            </div>

            <div className="space-y-8">
                {/* Connection Settings */}
                <section className="bg-matter/30 border border-white/5 rounded-xl p-8">
                    <div className="flex items-center gap-3 mb-6">
                        <Server className="text-cyan" size={24} />
                        <h2 className="text-xl font-bold text-white">Conexão Backend</h2>
                    </div>

                    <div className="space-y-4">
                        <div>
                            <label className="block text-xs font-mono text-white/40 uppercase mb-2">API Endpoint</label>
                            <input
                                type="text"
                                defaultValue="http://localhost:8000"
                                className="w-full bg-black/20 border border-white/10 rounded px-4 py-3 text-white focus:border-cyan/50 focus:outline-none transition-colors"
                            />
                        </div>
                        <div>
                            <label className="block text-xs font-mono text-white/40 uppercase mb-2">WebSocket Stream</label>
                            <input
                                type="text"
                                defaultValue="ws://localhost:8000/ws"
                                className="w-full bg-black/20 border border-white/10 rounded px-4 py-3 text-white focus:border-cyan/50 focus:outline-none transition-colors"
                            />
                        </div>
                    </div>
                </section>

                {/* Interface Settings */}
                <section className="bg-matter/30 border border-white/5 rounded-xl p-8">
                    <div className="flex items-center gap-3 mb-6">
                        <Monitor className="text-purple-400" size={24} />
                        <h2 className="text-xl font-bold text-white">Interface</h2>
                    </div>

                    <div className="grid grid-cols-2 gap-6">
                        <div>
                            <label className="block text-xs font-mono text-white/40 uppercase mb-2">Tema</label>
                            <select className="w-full bg-black/20 border border-white/10 rounded px-4 py-3 text-white focus:border-cyan/50 focus:outline-none appearance-none">
                                <option>Dark Mode (Padrão)</option>
                                <option>Light Mode</option>
                                <option>Cyberpunk</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-xs font-mono text-white/40 uppercase mb-2">Idioma</label>
                            <select className="w-full bg-black/20 border border-white/10 rounded px-4 py-3 text-white focus:border-cyan/50 focus:outline-none appearance-none">
                                <option>Português (BR)</option>
                                <option>English (US)</option>
                            </select>
                        </div>
                    </div>
                </section>

                <div className="flex justify-end pt-4">
                    <button className="bg-cyan text-black font-bold py-3 px-8 rounded-lg flex items-center gap-2 hover:bg-cyan/90 transition-colors">
                        <Save size={18} /> Salvar Alterações
                    </button>
                </div>
            </div>
        </div>
    );
};
