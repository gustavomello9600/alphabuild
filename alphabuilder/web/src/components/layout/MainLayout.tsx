import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const MainLayout = ({ children }: { children: React.ReactNode }) => {
    const [collapsed, setCollapsed] = useState(false);

    return (
        <div className="flex h-screen bg-void text-text font-sans selection:bg-cyan/30 selection:text-cyan overflow-hidden">
            <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />
            <div className="flex-1 flex flex-col relative min-w-0">
                <Header collapsed={collapsed} />
                <main className="flex-1 overflow-y-auto pt-16 p-8 relative">
                    {children}
                </main>
            </div>
        </div>
    );
};
