# End-to-End Tests for Web Interface

## Overview

This directory contains comprehensive end-to-end tests for the Robotics Model Optimization Platform web interface. These tests verify the complete user workflows and ensure all requirements are met.

## Test Coverage

### Requirements Covered

#### Requirement 5.1: Real-time Progress Monitoring
- ✅ Displays real-time optimization progress on dashboard
- ✅ Shows active optimizations with progress indicators
- ✅ Subscribes to WebSocket updates for live progress tracking
- ✅ Displays multiple concurrent optimizations

#### Requirement 5.2: Control Operations
- ✅ Allows users to pause running optimizations
- ✅ Allows users to resume paused optimizations
- ✅ Allows users to cancel optimizations
- ✅ Provides configuration interface for optimization criteria
- ✅ Validates and saves configuration updates

#### Requirement 5.3: Monitoring and History
- ✅ Displays optimization history with detailed logs
- ✅ Shows performance comparisons between original and optimized models
- ✅ Provides filtering and searching capabilities
- ✅ Displays notifications for optimization completion
- ✅ Handles errors gracefully with user feedback
- ✅ Allows access to detailed session information

## Test Files

### complete-workflow.e2e.test.tsx

Comprehensive end-to-end tests covering:

1. **Real-time Progress Monitoring**
   - Dashboard displays active optimizations
   - WebSocket subscriptions for progress updates
   - Progress bars and status indicators

2. **Control Operations**
   - Pause/Resume/Cancel functionality
   - Configuration management
   - User interaction flows

3. **Optimization History**
   - Session listing and filtering
   - Detailed session views
   - Performance metrics display

4. **Error Handling**
   - Network error recovery
   - User feedback messages
   - Graceful degradation

5. **Complete Workflows**
   - Navigation between pages
   - Data persistence across views
   - API integration verification

## Running the Tests

```bash
# Run all e2e tests
npm test -- --testPathPattern=e2e

# Run with coverage
npm test -- --testPathPattern=e2e --coverage

# Run in watch mode (for development)
npm test -- --testPathPattern=e2e --watch
```

## Test Structure

Each test follows this pattern:

1. **Setup**: Mock API responses and initial state
2. **Render**: Render the application
3. **Interact**: Simulate user interactions
4. **Assert**: Verify expected outcomes
5. **Cleanup**: Reset mocks and state

## Mocking Strategy

- **API Service**: All API calls are mocked using Jest
- **WebSocket**: WebSocket context is mocked to simulate real-time updates
- **Ant Design**: matchMedia is mocked for responsive components
- **Messages**: Ant Design message notifications are mocked for verification

## Key Features Tested

### Dashboard
- Statistics display (total models, active optimizations, etc.)
- Active optimizations table
- Performance charts
- Real-time updates via WebSocket

### Optimization History
- Session listing with filters
- Search functionality
- Status indicators
- Detailed session modal
- Performance metrics visualization

### Configuration
- Optimization criteria form
- Save functionality
- Validation feedback

### Error Handling
- Network error messages
- Retry mechanisms
- User feedback

## Test Data

Mock data includes:
- Optimization sessions (running, completed, failed)
- Dashboard statistics
- Optimization criteria
- Performance metrics
- Evaluation reports

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- No external dependencies required
- All network calls are mocked
- Deterministic test results
- Fast execution time

## Future Enhancements

Potential additions:
- Visual regression testing
- Performance benchmarking
- Accessibility testing
- Cross-browser testing
- Mobile responsiveness testing

## Troubleshooting

### Common Issues

1. **Timeout Errors**: Increase timeout in waitFor calls
2. **Element Not Found**: Check if element is rendered conditionally
3. **Mock Not Working**: Verify mock is set up before render
4. **Flaky Tests**: Add proper wait conditions

### Debug Tips

```typescript
// Add debug output
screen.debug();

// Check what's rendered
console.log(screen.getByRole('table'));

// Increase timeout for specific assertions
await waitFor(() => {
  expect(element).toBeInTheDocument();
}, { timeout: 5000 });
```

## Maintenance

- Update mocks when API contracts change
- Add tests for new features
- Remove tests for deprecated features
- Keep test data realistic and up-to-date
