import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Float, Octahedron, Icosahedron } from '@react-three/drei';
import { Copy, Check, Lock, ArrowRight, Activity, Cpu, Layers, AlertTriangle, BarChart2, Grid, Zap, Box, Move, Maximize, Shield, MousePointer, ToggleLeft, ToggleRight, CheckSquare, Square } from 'lucide-react';

// --- Components for the Design System Page ---

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <section className="mb-32">
        <div className="flex items-center gap-4 mb-12 border-b border-steel/30 pb-6">
            <h2 className="text-4xl font-display font-bold text-white tracking-tight">
                {title}
            </h2>
            <div className="h-px flex-1 bg-gradient-to-r from-steel/50 to-transparent" />
        </div>
        {children}
    </section>
);

const ColorCard = ({ name, hex, usage }: { name: string; hex: string; usage: string }) => {
    const [copied, setCopied] = useState(false);

    const copyToClipboard = () => {
        navigator.clipboard.writeText(hex);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <motion.div
            whileHover={{ y: -5 }}
            onClick={copyToClipboard}
            className="cursor-pointer group"
        >
            <div
                className="h-28 rounded-lg mb-4 border border-white/5 shadow-2xl relative overflow-hidden transition-all group-hover:border-white/20"
                style={{ backgroundColor: hex }}
            >
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/40 backdrop-blur-[2px]">
                    {copied ? <Check className="text-white" /> : <Copy className="text-white" />}
                </div>
            </div>
            <div>
                <div className="flex justify-between items-baseline mb-1">
                    <span className="font-display font-bold text-white text-lg">{name}</span>
                    <span className="font-mono text-xs text-white/40 uppercase">{hex}</span>
                </div>
                <p className="text-xs text-white/60 leading-relaxed">{usage}</p>
            </div>
        </motion.div>
    );
};

const TypographySample = ({ font, name, usage, sample }: { font: string; name: string; usage: string; sample?: React.ReactNode }) => (
    <div className="mb-12 p-8 border border-steel/30 rounded-xl bg-gradient-to-br from-matter to-deep">
        <div className="flex flex-col md:flex-row justify-between md:items-end mb-8 border-b border-white/5 pb-6 gap-4">
            <div>
                <h3 className="text-2xl text-white font-bold mb-1">{name}</h3>
                <p className="text-sm text-cyan font-medium flex items-center gap-2">
                    <Zap size={12} /> {usage}
                </p>
            </div>
            <span className="text-xs text-white/20 font-mono border border-white/10 px-2 py-1 rounded">{font}</span>
        </div>
        <div className={`${font} text-white`}>
            {sample || (
                <>
                    <p className="text-5xl md:text-7xl mb-6 tracking-tight font-medium">AlphaBuilder</p>
                    <p className="text-2xl md:text-3xl mb-6 text-white/80 leading-tight">
                        Otimização topológica guiada por inteligência artificial.
                    </p>
                    <p className="text-base text-white/50 max-w-2xl leading-relaxed">
                        A engenharia não é apenas sobre construir; é sobre esculpir o necessário e remover o supérfluo.
                        Nossa IA analisa milhões de possibilidades estruturais para encontrar a forma perfeita.
                    </p>
                </>
            )}
        </div>
    </div>
);

// --- 3D Logo Component (Octahedron) ---

const Logo3D = () => {
    return (
        <Float speed={2} rotationIntensity={0.5} floatIntensity={1}>
            <group scale={1.2}>
                {/* Core Octahedron */}
                <Octahedron args={[1, 0]}>
                    <meshStandardMaterial
                        color="#050505"
                        metalness={0.9}
                        roughness={0.1}
                        emissive="#7000FF"
                        emissiveIntensity={0.1}
                    />
                </Octahedron>

                {/* Wireframe Overlay */}
                <Octahedron args={[1.02, 0]}>
                    <meshBasicMaterial color="#00F0FF" wireframe transparent opacity={0.3} />
                </Octahedron>

                {/* Inner Core */}
                <Icosahedron args={[0.4, 0]}>
                    <meshStandardMaterial color="#FF0055" emissive="#FF0055" emissiveIntensity={2} toneMapped={false} />
                </Icosahedron>

                {/* Orbiting Rings */}
                <group rotation={[1, 0.5, 0]}>
                    <mesh>
                        <torusGeometry args={[1.8, 0.01, 16, 100]} />
                        <meshBasicMaterial color="#00F0FF" transparent opacity={0.2} />
                    </mesh>
                </group>
                <group rotation={[-0.5, 1, 0]}>
                    <mesh>
                        <torusGeometry args={[2.2, 0.01, 16, 100]} />
                        <meshBasicMaterial color="#7000FF" transparent opacity={0.2} />
                    </mesh>
                </group>
            </group>
        </Float>
    );
};

// --- Main Page ---

export default function DesignSystem() {
    return (
        <div className="min-h-screen bg-void text-text p-8 md:p-20 selection:bg-cyan/30 selection:text-cyan">

            {/* Hero Section */}
            <header className="flex flex-col lg:flex-row items-center justify-between mb-40 gap-12">
                <div className="lg:w-1/2 z-10">
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 1 }}
                    >
                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-cyan text-xs font-mono mb-6 backdrop-blur-sm">
                            <span className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
                            SYSTEM_V2.0
                        </div>
                        <h1 className="text-7xl md:text-9xl font-display font-bold text-white mb-8 tracking-tighter leading-[0.9]">
                            Alpha<br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan via-purple to-magenta">Design</span>
                        </h1>
                        <p className="text-xl md:text-2xl text-white/60 max-w-xl leading-relaxed font-light">
                            O sistema de design que define a intersecção entre <strong className="text-white font-medium">precisão estrutural</strong> e <strong className="text-white font-medium">inteligência generativa</strong>.
                        </p>
                    </motion.div>
                </div>

                <div className="lg:w-1/2 h-[500px] w-full relative">
                    <div className="absolute inset-0 bg-gradient-to-tr from-cyan/10 via-purple/10 to-magenta/10 blur-[120px] rounded-full opacity-40" />
                    <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
                        <ambientLight intensity={0.2} />
                        <pointLight position={[10, 10, 10]} color="#E0E0E0" intensity={1} />
                        <pointLight position={[-10, -5, -10]} color="#00F0FF" intensity={2} />
                        <pointLight position={[0, -10, 5]} color="#FF0055" intensity={1} />
                        <Logo3D />
                        <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.5} />
                    </Canvas>
                </div>
            </header>

            {/* 01. Colors */}
            <Section title="01. Cromática & Energia">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                    <div>
                        <h3 className="text-xs font-mono text-cyan mb-6 uppercase tracking-[0.2em] flex items-center gap-2">
                            <div className="w-8 h-[1px] bg-cyan" /> Base (60%)
                        </h3>
                        <div className="grid gap-2">
                            <ColorCard name="Preto Vácuo" hex="#050505" usage="Background principal. A ausência de luz onde a estrutura nasce." />
                            <ColorCard name="Espaço Profundo" hex="#0A0A0A" usage="Camadas secundárias e profundidade de campo." />
                        </div>
                    </div>
                    <div>
                        <h3 className="text-xs font-mono text-white/40 mb-6 uppercase tracking-[0.2em] flex items-center gap-2">
                            <div className="w-8 h-[1px] bg-white/40" /> Estrutura (30%)
                        </h3>
                        <div className="grid gap-2">
                            <ColorCard name="Cinza Material" hex="#121212" usage="Superfícies de interface, cards e painéis." />
                            <ColorCard name="Aço Estrutural" hex="#2A2A2A" usage="Bordas, divisores e elementos desabilitados." />
                            <ColorCard name="Titânio" hex="#E0E0E0" usage="Texto primário e ícones de alta visibilidade." />
                        </div>
                    </div>
                    <div>
                        <h3 className="text-xs font-mono text-magenta mb-6 uppercase tracking-[0.2em] flex items-center gap-2">
                            <div className="w-8 h-[1px] bg-magenta" /> Energia (10%)
                        </h3>
                        <div className="grid gap-2">
                            <ColorCard name="Ciano: Contorno" hex="#00F0FF" usage="Ações primárias, condições de contorno fixas, segurança." />
                            <ColorCard name="Laranja: Carga" hex="#FF6B00" usage="Cargas aplicadas, forças externas, pontos de aplicação." />
                            <ColorCard name="Magenta: Remoção" hex="#FF0055" usage="Ações de remoção, policy remove, alertas críticos." />
                            <ColorCard name="Roxo: Neural" hex="#7000FF" usage="Visualização de IA, gradientes de raciocínio." />
                        </div>
                    </div>
                </div>
            </Section>

            {/* 02. Typography */}
            <Section title="02. Tipografia">
                <TypographySample
                    font="font-display"
                    name="Space Grotesk"
                    usage="Identidade & Impacto"
                />
                <TypographySample
                    font="font-sans"
                    name="Inter"
                    usage="Interface & Leitura"
                    sample={
                        <div className="space-y-4">
                            <p className="text-lg text-white/80">
                                A tipografia funcional deve ser invisível até que seja necessária.
                                Usamos a Inter para garantir legibilidade máxima em densidades altas de dados.
                            </p>
                            <div className="grid grid-cols-2 gap-4 text-sm text-white/50 font-mono border-t border-white/10 pt-4">
                                <span>Peso: Regular (400)</span>
                                <span>Tracking: -0.01em</span>
                                <span>Peso: Medium (500)</span>
                                <span>Line-height: 1.5</span>
                            </div>
                        </div>
                    }
                />
                <TypographySample
                    font="font-mono"
                    name="JetBrains Mono"
                    usage="Dados & Código"
                    sample={
                        <div className="bg-black/50 p-4 rounded border border-white/10 text-sm text-cyan/80">
                            <p>def optimize_topology(mesh, loads):</p>
                            <p className="pl-4">density = initialize_density(mesh)</p>
                            <p className="pl-4">while not converged:</p>
                            <p className="pl-8">compliance = fea_solve(density, loads)</p>
                            <p className="pl-8">density = update_gradient(compliance)</p>
                            <p className="pl-4">return density</p>
                        </div>
                    }
                />
            </Section>

            {/* 03. Spacing & Layout */}
            <Section title="03. Espaçamento & Layout">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
                    {/* Spacing Scale */}
                    <div>
                        <h3 className="text-xl font-display text-white mb-8 flex items-center gap-2">
                            <Maximize size={20} className="text-cyan" />
                            Escala de Espaçamento
                        </h3>
                        <div className="space-y-4">
                            {[4, 8, 12, 16, 24, 32, 48, 64].map((size) => (
                                <div key={size} className="flex items-center gap-4">
                                    <span className="w-12 text-xs font-mono text-white/40 text-right">{size}px</span>
                                    <div className="h-8 bg-cyan/20 border border-cyan/40 rounded flex items-center justify-center" style={{ width: size }}>
                                        <span className="text-[10px] text-cyan opacity-0 hover:opacity-100">{size}</span>
                                    </div>
                                    <span className="text-xs text-white/30">space-{size / 4}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Grid System */}
                    <div>
                        <h3 className="text-xl font-display text-white mb-8 flex items-center gap-2">
                            <Grid size={20} className="text-cyan" />
                            Grid de 12 Colunas
                        </h3>
                        <div className="p-6 border border-steel/30 rounded-xl bg-matter/30">
                            <div className="grid grid-cols-12 gap-2 h-32 w-full relative">
                                {Array.from({ length: 12 }).map((_, i) => (
                                    <div key={i} className="bg-cyan/5 border border-cyan/20 rounded flex flex-col items-center justify-center text-xs text-cyan/50 font-mono group hover:bg-cyan/10 transition-colors">
                                        <span className="hidden md:block">{i + 1}</span>
                                    </div>
                                ))}
                            </div>
                            <div className="mt-4 flex justify-between text-xs font-mono text-white/40">
                                <span>Gutter: 1rem (16px)</span>
                                <span>Max-Width: 1440px</span>
                            </div>
                        </div>
                    </div>
                </div>
            </Section>

            {/* 04. Shape & Surface */}
            <Section title="04. Forma & Superfície">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                    <div>
                        <h3 className="text-lg font-display text-white mb-6 flex items-center gap-2">
                            <Box size={20} className="text-purple" />
                            Border Radius
                        </h3>
                        <div className="space-y-4">
                            <div className="p-4 border border-white/10 bg-white/5 rounded-sm text-center text-xs text-white/50">sm (2px)</div>
                            <div className="p-4 border border-white/10 bg-white/5 rounded text-center text-xs text-white/50">default (4px)</div>
                            <div className="p-4 border border-white/10 bg-white/5 rounded-lg text-center text-xs text-white/50">lg (8px)</div>
                            <div className="p-4 border border-white/10 bg-white/5 rounded-xl text-center text-xs text-white/50">xl (12px)</div>
                            <div className="p-4 border border-white/10 bg-white/5 rounded-full text-center text-xs text-white/50">full (9999px)</div>
                        </div>
                    </div>

                    <div>
                        <h3 className="text-lg font-display text-white mb-6 flex items-center gap-2">
                            <Layers size={20} className="text-purple" />
                            Elevação & Sombras
                        </h3>
                        <div className="space-y-6">
                            <div className="h-16 bg-matter border border-steel rounded flex items-center justify-center text-xs text-white/50 shadow-sm">Shadow SM</div>
                            <div className="h-16 bg-matter border border-steel rounded flex items-center justify-center text-xs text-white/50 shadow-lg">Shadow LG</div>
                            <div className="h-16 bg-matter border border-steel rounded flex items-center justify-center text-xs text-white/50 shadow-[0_0_30px_rgba(0,240,255,0.15)] border-cyan/30">Neon Glow</div>
                        </div>
                    </div>

                    <div>
                        <h3 className="text-lg font-display text-white mb-6 flex items-center gap-2">
                            <Shield size={20} className="text-purple" />
                            Bordas & Strokes
                        </h3>
                        <div className="space-y-4">
                            <div className="h-12 border border-white/10 rounded flex items-center justify-center text-xs text-white/50">1px Solid (Subtle)</div>
                            <div className="h-12 border border-white/30 rounded flex items-center justify-center text-xs text-white/50">1px Solid (Active)</div>
                            <div className="h-12 border border-dashed border-white/30 rounded flex items-center justify-center text-xs text-white/50">1px Dashed</div>
                            <div className="h-12 border-l-4 border-cyan bg-cyan/5 rounded-r flex items-center justify-center text-xs text-white/50">Left Accent</div>
                        </div>
                    </div>
                </div>
            </Section>

            {/* 05. Iconography */}
            <Section title="05. Iconografia">
                <p className="text-white/60 mb-8 max-w-2xl">
                    Utilizamos a biblioteca <strong>Lucide React</strong> com traços finos (1.5px ou 2px) para manter a elegância técnica.
                    Ícones devem ser usados para reforçar o significado, não apenas para decoração.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
                    {[Activity, Cpu, Layers, AlertTriangle, BarChart2, Grid, Zap, Box, Move, Maximize, Shield, MousePointer, Lock, Copy, Check, ArrowRight].map((Icon, i) => (
                        <div key={i} className="aspect-square border border-white/5 bg-white/5 rounded flex flex-col items-center justify-center gap-2 hover:border-cyan/50 hover:text-cyan transition-colors group">
                            <Icon size={24} strokeWidth={1.5} />
                        </div>
                    ))}
                </div>
            </Section>

            {/* 06. Components */}
            <Section title="06. Biblioteca de Componentes">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">

                    {/* Interactive Elements */}
                    <div className="space-y-12">

                        {/* Buttons */}
                        <div className="space-y-4">
                            <h3 className="text-lg font-display text-white mb-4">Botões</h3>
                            <div className="flex flex-wrap gap-4">
                                <button className="px-6 py-3 bg-cyan text-black font-bold rounded hover:bg-white hover:shadow-[0_0_20px_rgba(0,240,255,0.4)] transition-all flex items-center gap-2 transform hover:-translate-y-0.5 active:translate-y-0">
                                    <Activity size={18} strokeWidth={2.5} />
                                    Primário
                                </button>
                                <button className="px-6 py-3 border border-white/20 text-white rounded hover:border-cyan hover:text-cyan hover:bg-cyan/5 transition-all flex items-center gap-2">
                                    <Lock size={18} />
                                    Secundário
                                </button>
                                <button className="px-6 py-3 text-white/60 hover:text-white transition-all flex items-center gap-2">
                                    Terciário
                                </button>
                            </div>
                        </div>

                        {/* Toggles & Checks */}
                        <div className="space-y-4">
                            <h3 className="text-lg font-display text-white mb-4">Seleção & Toggles</h3>
                            <div className="flex items-center gap-8 p-6 border border-steel/30 rounded-xl bg-matter/30">
                                <div className="flex items-center gap-2 text-white/80">
                                    <ToggleRight size={32} className="text-cyan" />
                                    <span className="text-sm">Ativado</span>
                                </div>
                                <div className="flex items-center gap-2 text-white/50">
                                    <ToggleLeft size={32} />
                                    <span className="text-sm">Desativado</span>
                                </div>
                                <div className="flex items-center gap-2 text-white/80">
                                    <CheckSquare size={20} className="text-cyan" />
                                    <span className="text-sm">Checkbox</span>
                                </div>
                                <div className="flex items-center gap-2 text-white/50">
                                    <Square size={20} />
                                    <span className="text-sm">Vazio</span>
                                </div>
                            </div>
                        </div>

                        {/* Inputs */}
                        <div className="space-y-4">
                            <h3 className="text-lg font-display text-white mb-4">Inputs</h3>
                            <div className="grid gap-4">
                                <div className="relative group">
                                    <input
                                        type="text"
                                        placeholder="Digite um valor..."
                                        className="w-full bg-black/50 border border-steel rounded p-3 text-white focus:border-cyan focus:outline-none focus:shadow-[0_0_15px_rgba(0,240,255,0.1)] transition-all font-mono peer"
                                    />
                                </div>
                                <div className="relative">
                                    <select className="w-full bg-black/50 border border-steel rounded p-3 text-white focus:border-purple focus:outline-none appearance-none font-mono cursor-pointer hover:border-white/30 transition-colors">
                                        <option>Opção A</option>
                                        <option>Opção B</option>
                                    </select>
                                    <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-white/50">
                                        <Layers size={16} />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Cards & Feedback */}
                    <div className="space-y-12">

                        {/* Badges */}
                        <div className="space-y-4">
                            <h3 className="text-lg font-display text-white mb-4">Badges & Tags</h3>
                            <div className="flex flex-wrap gap-4">
                                <span className="px-3 py-1 rounded-full bg-cyan/10 text-cyan border border-cyan/20 text-xs font-mono">STATUS: ONLINE</span>
                                <span className="px-3 py-1 rounded-full bg-purple/10 text-purple border border-purple/20 text-xs font-mono">IA: ATIVA</span>
                                <span className="px-3 py-1 rounded-full bg-magenta/10 text-magenta border border-magenta/20 text-xs font-mono">ERRO CRÍTICO</span>
                                <span className="px-3 py-1 rounded-full bg-white/5 text-white/60 border border-white/10 text-xs font-mono">NEUTRO</span>
                            </div>
                        </div>

                        {/* Project Card */}
                        <div className="space-y-4">
                            <h3 className="text-lg font-display text-white mb-4">Cards Complexos</h3>
                            <div className="p-6 rounded-xl bg-gradient-to-b from-white/5 to-transparent border border-white/10 hover:border-cyan/50 transition-all cursor-pointer group hover:shadow-2xl hover:shadow-cyan/5">
                                <div className="flex justify-between items-start mb-6">
                                    <div className="p-3 bg-purple/10 rounded-lg text-purple group-hover:text-cyan group-hover:bg-cyan/10 transition-colors border border-purple/20 group-hover:border-cyan/20">
                                        <Cpu size={24} />
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                                        <span className="text-xs font-mono text-white/40">ONLINE</span>
                                    </div>
                                </div>
                                <h4 className="text-xl font-bold text-white mb-2 group-hover:text-cyan transition-colors">Estudo Topológico #8F3A</h4>
                                <p className="text-sm text-white/60 mb-6 leading-relaxed">
                                    Refinamento estrutural de viga em balanço com restrição de volume de 40%.
                                </p>
                                <div className="flex items-center justify-between border-t border-white/5 pt-4">
                                    <span className="text-xs font-mono text-white/30">ATUALIZADO HÁ 2M</span>
                                    <div className="flex items-center text-cyan text-sm font-medium gap-1 group-hover:translate-x-1 transition-transform">
                                        Acessar Dados <ArrowRight size={14} />
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </Section>

            {/* 07. Data Viz */}
            <Section title="07. Visualização de Dados">
                <div className="p-8 border border-steel/30 rounded-xl bg-matter/30">
                    <div className="flex justify-between items-end mb-8">
                        <h3 className="text-lg font-display text-white flex items-center gap-2">
                            <BarChart2 size={20} className="text-purple" />
                            Convergência de Otimização
                        </h3>
                        <div className="flex gap-4 text-xs font-mono text-white/40">
                            <span className="flex items-center gap-2"><div className="w-2 h-2 bg-purple rounded-full" /> Massa</span>
                            <span className="flex items-center gap-2"><div className="w-2 h-2 bg-cyan rounded-full" /> Rigidez</span>
                        </div>
                    </div>

                    <div className="relative h-64 w-full border-l border-b border-white/10">
                        {/* Grid Lines */}
                        <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
                            {[100, 75, 50, 25, 0].map((val) => (
                                <div key={val} className="w-full h-px bg-white/5 relative">
                                    <span className="absolute -left-8 -top-2 text-[10px] font-mono text-white/30">{val}%</span>
                                </div>
                            ))}
                        </div>

                        {/* Bars */}
                        <div className="absolute inset-0 flex items-end justify-around px-4 pt-4">
                            {[45, 72, 58, 92, 35, 64, 88, 50, 78, 95].map((h, i) => (
                                <div key={i} className="w-full mx-1 flex flex-col justify-end group h-full relative">
                                    <div
                                        className="w-full bg-gradient-to-t from-purple/20 to-purple rounded-t hover:from-cyan/20 hover:to-cyan transition-all duration-300 relative"
                                        style={{ height: `${h}%` }}
                                    >
                                        <div className="absolute -top-10 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-all transform translate-y-2 group-hover:translate-y-0 bg-black border border-white/20 px-2 py-1 rounded text-xs text-white font-mono z-10 whitespace-nowrap">
                                            Iteração {i + 1}: {h}%
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </Section>

            {/* 09. Motion */}
            <Section title="09. Motion & Animação">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    <div className="p-6 border border-steel/30 rounded-xl bg-matter/30">
                        <h3 className="text-sm font-mono text-white/60 mb-4 uppercase">Linear (Mecânico)</h3>
                        <div className="h-32 flex items-center justify-center bg-black/20 rounded-lg mb-4 overflow-hidden relative">
                            <motion.div
                                animate={{ x: [-50, 50] }}
                                transition={{ repeat: Infinity, repeatType: "reverse", duration: 1, ease: "linear" }}
                                className="w-12 h-12 bg-steel rounded-full"
                            />
                        </div>
                        <p className="text-xs text-white/40">Uso: Loaders, rotações contínuas.</p>
                    </div>

                    <div className="p-6 border border-steel/30 rounded-xl bg-matter/30">
                        <h3 className="text-sm font-mono text-white/60 mb-4 uppercase">Ease Out (Natural)</h3>
                        <div className="h-32 flex items-center justify-center bg-black/20 rounded-lg mb-4 overflow-hidden relative">
                            <motion.div
                                animate={{ x: [-50, 50] }}
                                transition={{ repeat: Infinity, repeatType: "reverse", duration: 1, ease: "easeOut" }}
                                className="w-12 h-12 bg-cyan rounded-full shadow-[0_0_15px_rgba(0,240,255,0.5)]"
                            />
                        </div>
                        <p className="text-xs text-white/40">Uso: Entradas de UI, Modais, Tooltips.</p>
                    </div>

                    <div className="p-6 border border-steel/30 rounded-xl bg-matter/30">
                        <h3 className="text-sm font-mono text-white/60 mb-4 uppercase">Elastic (Energético)</h3>
                        <div className="h-32 flex items-center justify-center bg-black/20 rounded-lg mb-4 overflow-hidden relative">
                            <motion.div
                                animate={{ scale: [0.8, 1.2] }}
                                transition={{ repeat: Infinity, repeatType: "reverse", duration: 1.5, type: "spring", bounce: 0.6 }}
                                className="w-12 h-12 bg-magenta rounded-full shadow-[0_0_15px_rgba(255,0,85,0.5)]"
                            />
                        </div>
                        <p className="text-xs text-white/40">Uso: Notificações, Erros, Ações de Sucesso.</p>
                    </div>
                </div>
            </Section>

            {/* 10. Analysis */}
            <Section title="10. Status do Sistema">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div className="p-6 border border-steel rounded-lg bg-matter/30">
                        <h3 className="text-lg font-display text-white mb-4 flex items-center gap-2">
                            <Check className="text-green-400" size={20} />
                            Sistema Completo
                        </h3>
                        <ul className="space-y-2 text-white/70 text-sm">
                            <li>• <strong>Identidade:</strong> Logo Octaedro, Cores, Tipografia.</li>
                            <li>• <strong>Fundamentos:</strong> Grid, Espaçamento, Radius, Sombras.</li>
                            <li>• <strong>Componentes:</strong> Botões, Inputs, Cards, Toggles, Badges.</li>
                            <li>• <strong>Iconografia:</strong> Set Lucide integrado.</li>
                            <li>• <strong>Data Viz:</strong> Gráficos científicos base.</li>
                            <li>• <strong>Motion:</strong> Curvas de animação definidas.</li>
                        </ul>
                    </div>

                    <div className="p-6 border border-cyan/30 rounded-lg bg-cyan/5">
                        <h3 className="text-lg font-display text-white mb-4 flex items-center gap-2">
                            <Activity className="text-cyan" size={20} />
                            Pronto para Implementação
                        </h3>
                        <p className="text-sm text-white/70">
                            O Design System "Computational Ethereal" atingiu maturidade suficiente para sustentar a construção da aplicação principal.
                        </p>
                    </div>
                </div>
            </Section>

        </div>
    );
}
