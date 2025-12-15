import '@testing-library/jest-dom';

// ResizeObserver polyfill
(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() { }
    unobserve() { }
    disconnect() { }
};
