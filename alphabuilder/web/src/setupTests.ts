import '@testing-library/jest-dom';
import { vi } from 'vitest';

// ResizeObserver polyfill
global.ResizeObserver = class ResizeObserver {
    observe() { }
    unobserve() { }
    disconnect() { }
};
