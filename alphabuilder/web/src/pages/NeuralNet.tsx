import { Cpu, Activity, Zap, GitBranch, Layers } from 'lucide-react';

const MetricCard = ({ label, value, unit, change, icon: Icon }: any) => (
    <div className="bg-matter/30 border border-white/5 p-6 rounded-xl relative overflow-hidden group hover:border-cyan/30 transition-colors">
        <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Icon size={48} />
        </div>
        <div className="text-white/40 text-sm font-mono mb-2 uppercase">{label}</div>
        <div className="text-3xl font-display font-bold text-white mb-1">
            {value} <span className="text-sm text-white/40 font-normal">{unit}</span>
        </div>
        <div className={`text-xs font-mono ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {change > 0 ? '+' : ''}{change}% vs last epoch
        </div>
    </div>
);

export const NeuralNet = () => {
    return (
        <div className="max-w-7xl mx-auto">
            <div className="mb-12">
                <h1 className="text-4xl font-display font-bold text-white mb-2">
                    Rede <span className="text-purple-400">Neural</span>
                </h1>
                <p className="text-white/60 max-w-xl">
                    Monitoramento em tempo real da arquitetura Swin-UNETR e métricas de treinamento.
                </p>
            </div>

            {/* Status Banner */}
            <div className="bg-gradient-to-r from-purple-900/20 to-cyan/10 border border-white/10 rounded-xl p-8 mb-8 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400">
                        <Cpu size={24} />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-white">Swin-UNETR (Physics-Aware)</h3>
                        <p className="text-white/40 text-sm">Versão 1.1.0 • Checkpoint #420</p>
                    </div>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-black/40 rounded-lg border border-white/10">
                    <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                    <span className="text-sm font-mono text-yellow-500">STANDBY (MOCK)</span>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
                <MetricCard label="Loss Total" value="0.042" unit="" change={-12.5} icon={Activity} />
                <MetricCard label="Acurácia (Policy)" value="94.2" unit="%" change={+2.1} icon={Zap} />
                <MetricCard label="Value Error" value="0.015" unit="MSE" change={-5.4} icon={GitBranch} />
                <MetricCard label="Inferência" value="45" unit="ms" change={-1.2} icon={Layers} />
            </div>

            {/* Architecture Diagram Placeholder */}
            <div className="bg-matter/30 border border-white/5 rounded-xl p-8 h-96 flex flex-col items-center justify-center text-center relative overflow-hidden">
                <div className="absolute inset-0 opacity-20 bg-[url('https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Neural_network.svg/1200px-Neural_network.svg.png')] bg-center bg-no-repeat bg-contain filter grayscale" />
                <div className="relative z-10">
                    <h3 className="text-xl font-bold text-white mb-2">Visualização da Arquitetura</h3>
                    <p className="text-white/40 max-w-md mx-auto">
                        O modelo utiliza um encoder Swin Transformer hierárquico para capturar características globais e locais da topologia 3D.
                    </p>
                </div>
            </div>
        </div>
    );
};
