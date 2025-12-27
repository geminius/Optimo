import '@testing-library/jest-dom';

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock matchMedia - required for Ant Design responsive components
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  configurable: true,
  value: jest.fn().mockImplementation(query => {
    const listeners: Array<(e: any) => void> = [];
    const mediaQueryList = {
      matches: query === '(min-width: 768px)', // Default to desktop
      media: query,
      onchange: null,
      addListener: jest.fn((listener: (e: any) => void) => {
        listeners.push(listener);
      }), // deprecated but still used by some libraries
      removeListener: jest.fn((listener: (e: any) => void) => {
        const index = listeners.indexOf(listener);
        if (index > -1) {
          listeners.splice(index, 1);
        }
      }), // deprecated
      addEventListener: jest.fn((event: string, listener: (e: any) => void) => {
        if (event === 'change') {
          listeners.push(listener);
        }
      }),
      removeEventListener: jest.fn((event: string, listener: (e: any) => void) => {
        if (event === 'change') {
          const index = listeners.indexOf(listener);
          if (index > -1) {
            listeners.splice(index, 1);
          }
        }
      }),
      dispatchEvent: jest.fn((event: Event) => {
        listeners.forEach(listener => listener(event));
        return true;
      }),
    };
    return mediaQueryList;
  }),
});

// Mock Ant Design's responsive observer breakpoints
const mockBreakpoints = {
  xs: '(max-width: 575px)',
  sm: '(min-width: 576px)',
  md: '(min-width: 768px)',
  lg: '(min-width: 992px)',
  xl: '(min-width: 1200px)',
  xxl: '(min-width: 1600px)',
};

// Ensure all breakpoints return proper MediaQueryList objects
Object.keys(mockBreakpoints).forEach(breakpoint => {
  const query = mockBreakpoints[breakpoint as keyof typeof mockBreakpoints];
  window.matchMedia(query);
});

// Mock window.screen for responsive observer
Object.defineProperty(window, 'screen', {
  writable: true,
  value: {
    width: 1024,
    height: 768,
  },
});

// Mock window.getComputedStyle
Object.defineProperty(window, 'getComputedStyle', {
  writable: true,
  value: jest.fn().mockImplementation(() => ({
    getPropertyValue: jest.fn((prop: string) => {
      // Mock CSS properties that Ant Design components might need
      const mockValues: Record<string, string> = {
        'scrollbar-color': 'auto',
        'scrollbar-width': 'auto',
        'overflow': 'visible',
        'display': 'block',
        'position': 'static',
        'width': '100px',
        'height': '100px',
      };
      return mockValues[prop] || '';
    }),
    // Add common CSS properties that might be accessed directly
    scrollbarColor: 'auto',
    scrollbarWidth: 'auto',
    overflow: 'visible',
    display: 'block',
    position: 'static',
    width: '100px',
    height: '100px',
  })),
});

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock as any;

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  log: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Mock Ant Design's message component
jest.mock('antd', () => {
  const antd = jest.requireActual('antd');
  return {
    ...antd,
    message: {
      success: jest.fn(),
      error: jest.fn(),
      info: jest.fn(),
      warning: jest.fn(),
      loading: jest.fn(),
    },
  };
});

// Mock Ant Design's responsive observer to prevent breakpoint issues
jest.mock('antd/lib/_util/responsiveObserve', () => ({
  __esModule: true,
  default: {
    subscribe: jest.fn(() => 'mock-token'),
    unsubscribe: jest.fn(),
    register: jest.fn(),
    responsiveMap: {
      xs: '(max-width: 575px)',
      sm: '(min-width: 576px)',
      md: '(min-width: 768px)',
      lg: '(min-width: 992px)',
      xl: '(min-width: 1200px)',
      xxl: '(min-width: 1600px)',
    },
  },
}));

// Mock recharts to avoid canvas issues in tests
jest.mock('recharts', () => {
  const React = require('react');
  return {
    LineChart: ({ children }: any) => React.createElement('div', { 'data-testid': 'line-chart' }, children),
    Line: () => React.createElement('div', { 'data-testid': 'line' }),
    XAxis: () => React.createElement('div', { 'data-testid': 'x-axis' }),
    YAxis: () => React.createElement('div', { 'data-testid': 'y-axis' }),
    CartesianGrid: () => React.createElement('div', { 'data-testid': 'cartesian-grid' }),
    Tooltip: () => React.createElement('div', { 'data-testid': 'tooltip' }),
    ResponsiveContainer: ({ children }: any) => React.createElement('div', { 'data-testid': 'responsive-container' }, children),
    BarChart: ({ children }: any) => React.createElement('div', { 'data-testid': 'bar-chart' }, children),
    Bar: () => React.createElement('div', { 'data-testid': 'bar' }),
  };
});

// Mock rc-table to prevent scrollbar measurement issues
jest.mock('rc-table', () => {
  const React = require('react');
  return {
    __esModule: true,
    default: ({ columns, dataSource, children }: any) => {
      return React.createElement('div', { 'data-testid': 'rc-table' }, 
        React.createElement('div', null, 'Mocked Table'),
        children
      );
    },
  };
});

// Mock rc-util's getScrollBarSize to prevent CSS issues
jest.mock('rc-util/lib/getScrollBarSize', () => ({
  __esModule: true,
  default: jest.fn(() => 17), // Standard scrollbar width
  getTargetScrollBarSize: jest.fn(() => 17), // Also mock the named export
}));

// Increase timeout for async operations
jest.setTimeout(10000);