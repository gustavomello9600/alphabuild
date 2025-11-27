import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const MainLayout = ({ children }: { children: React.ReactNode }) => {
    return (
        <div className="flex min-h-screen bg-void text-text font-sans selection:bg-cyan/30 selection:text-cyan overflow-hidden">
            <Sidebar />
            <div className="flex-1 flex flex-col relative">
                <Header />
                <main className="flex-1 overflow-y-auto pt-16 p-8 relative">
                    {children}
                </main>
            </div>
        </div>
    );
};
