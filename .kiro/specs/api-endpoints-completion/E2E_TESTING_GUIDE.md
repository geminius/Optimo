# End-to-End Frontend Integration Testing Guide

This guide provides step-by-step instructions for manually testing the frontend-backend integration for all implemented features.

## Prerequisites

1. **Backend Running**: Start the backend API server
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Frontend Running**: Start the frontend development server
   ```bash
   cd frontend && npm start
   ```

3. **Browser**: Open browser to `http://localhost:3000`

## Test 1: Dashboard Page Integration (Requirements 1.1-1.5)

### 1.1 Verify Statistics Load Correctly

**Steps:**
1. Navigate to the Dashboard page
2. Observe the statistics cards at the top

**Expected Results:**
- ✓ "Total Models" card displays a number
- ✓ "Active Optimizations" card displays a number
- ✓ "Avg Size Reduction" card displays a percentage
- ✓ "Avg Speed Improvement" card displays a percentage
- ✓ No error messages appear
- ✓ Loading spinners disappear after data loads

**API Call to Verify:**
```bash
curl -X GET http://localhost:8000/dashboard/stats \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 1.2 Verify Error Handling When Backend Unavailable

**Steps:**
1. Stop the backend server
2. Refresh the dashboard page
3. Observe error handling

**Expected Results:**
- ✓ Error message displays indicating connection failure
- ✓ Page doesn't crash or show blank screen
- ✓ User-friendly error message appears

**Steps to Restore:**
1. Restart the backend server
2. Refresh the page

### 1.3 Verify Data Refreshes on Page Reload

**Steps:**
1. Note the current statistics values
2. Upload a new model or start an optimization
3. Refresh the dashboard page
4. Observe updated statistics

**Expected Results:**
- ✓ Statistics update to reflect new data
- ✓ "Last Updated" timestamp changes
- ✓ No stale data is displayed

---

## Test 2: Sessions List Integration (Requirements 2.1-2.5)

### 2.1 Verify Sessions Display Correctly

**Steps:**
1. Navigate to the Optimization History page
2. Observe the sessions table

**Expected Results:**
- ✓ Table displays with columns: Session ID, Model, Status, Progress, Techniques, Actions
- ✓ Each session shows correct information
- ✓ Status badges have appropriate colors (blue=running, green=completed, red=failed)
- ✓ Progress bars show current progress percentage

**API Call to Verify:**
```bash
curl -X GET http://localhost:8000/optimization/sessions \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 2.2 Verify Filtering Works

**Steps:**
1. Use the status filter dropdown
2. Select "Completed"
3. Observe filtered results

**Expected Results:**
- ✓ Only completed sessions are displayed
- ✓ Total count updates to match filtered results
- ✓ Filter can be cleared to show all sessions

**Test Other Filters:**
- Filter by date range
- Filter by model ID
- Combine multiple filters

**API Call to Verify:**
```bash
curl -X GET "http://localhost:8000/optimization/sessions?status=completed" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 2.3 Verify Pagination Works

**Steps:**
1. If more than 10 sessions exist, observe pagination controls
2. Click "Next Page"
3. Observe new set of sessions

**Expected Results:**
- ✓ Pagination controls appear when needed
- ✓ Clicking next/previous loads new sessions
- ✓ Page numbers update correctly
- ✓ No duplicate sessions appear across pages

**API Call to Verify:**
```bash
curl -X GET "http://localhost:8000/optimization/sessions?skip=10&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 2.4 Verify Empty State

**Steps:**
1. Apply filters that return no results
2. Observe empty state display

**Expected Results:**
- ✓ "No sessions found" message displays
- ✓ Helpful text suggests trying different filters
- ✓ No error messages appear

---

## Test 3: Configuration Page Integration (Requirements 3.1-3.5)

### 3.1 Verify Configuration Loads Correctly

**Steps:**
1. Navigate to the Configuration page
2. Observe the form fields

**Expected Results:**
- ✓ All configuration fields are populated
- ✓ Current values are displayed correctly
- ✓ Form is editable
- ✓ No loading errors appear

**API Call to Verify:**
```bash
curl -X GET http://localhost:8000/config/optimization-criteria \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### 3.2 Verify Configuration Updates Save Properly

**Steps:**
1. Modify a configuration value (e.g., accuracy threshold)
2. Click "Save Configuration"
3. Observe success message
4. Refresh the page
5. Verify the updated value persists

**Expected Results:**
- ✓ Success message appears after saving
- ✓ Updated values persist after page refresh
- ✓ No errors during save operation

**API Call to Verify:**
```bash
curl -X PUT http://localhost:8000/config/optimization-criteria \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_config",
    "description": "Test configuration",
    "constraints": {
      "preserve_accuracy_threshold": 0.95,
      "allowed_techniques": ["quantization", "pruning"]
    },
    "target_deployment": "edge",
    "enabled_techniques": ["quantization", "pruning"]
  }'
```

### 3.3 Verify Validation Errors Display Correctly

**Steps:**
1. Enter an invalid value (e.g., accuracy threshold > 1.0)
2. Click "Save Configuration"
3. Observe validation error

**Expected Results:**
- ✓ Error message displays near the invalid field
- ✓ Error message is clear and actionable
- ✓ Form is not submitted
- ✓ User can correct the error and resubmit

**Test Invalid Values:**
- Accuracy threshold < 0 or > 1.0
- Empty required fields
- Invalid technique names
- Conflicting settings

### 3.4 Test Various Configuration Values

**Steps:**
1. Test aggressive optimization settings:
   - Low accuracy threshold (0.85)
   - High size reduction (70%)
   - Multiple techniques enabled

2. Test conservative optimization settings:
   - High accuracy threshold (0.98)
   - Low size reduction (20%)
   - Single technique enabled

**Expected Results:**
- ✓ All valid configurations save successfully
- ✓ Different presets can be switched between
- ✓ No data loss when switching configurations

---

## Test 4: WebSocket Real-Time Updates (Requirements 4.1-4.6)

### 4.1 Verify Connection Status Indicator

**Steps:**
1. Observe the connection status indicator (usually in header/footer)
2. Note "Connected" status

**Expected Results:**
- ✓ Status shows "Connected" with green indicator
- ✓ Connection establishes automatically on page load

### 4.2 Verify Progress Updates Appear in Real-Time

**Steps:**
1. Start a new optimization session
2. Navigate to Dashboard or History page
3. Observe the progress bar for the active session
4. Watch for real-time updates

**Expected Results:**
- ✓ Progress bar updates without page refresh
- ✓ Status changes from "running" to "completed" automatically
- ✓ Current step information updates in real-time
- ✓ Updates appear within 1-2 seconds of backend changes

**Monitor WebSocket Events:**
Open browser DevTools → Network → WS tab to see WebSocket messages

### 4.3 Verify Reconnection After Disconnect

**Steps:**
1. Open browser DevTools → Network tab
2. Note the WebSocket connection
3. Stop the backend server
4. Observe "Disconnected" status
5. Restart the backend server
6. Observe automatic reconnection

**Expected Results:**
- ✓ Status changes to "Disconnected" when backend stops
- ✓ Status changes to "Connecting..." when attempting reconnect
- ✓ Status changes to "Connected" after successful reconnect
- ✓ No manual refresh required
- ✓ Updates resume after reconnection

### 4.4 Test Multiple Browser Tabs

**Steps:**
1. Open the application in two browser tabs
2. Start an optimization in one tab
3. Observe updates in both tabs

**Expected Results:**
- ✓ Both tabs receive real-time updates
- ✓ Progress updates appear in both tabs simultaneously
- ✓ No conflicts or duplicate updates
- ✓ Each tab maintains its own WebSocket connection

---

## Test 5: Authentication Flows (Requirements 5.1-5.5)

### 5.1 Verify Login Flow

**Steps:**
1. Navigate to login page (if not logged in)
2. Enter credentials: username="admin", password="admin"
3. Click "Login"
4. Observe redirect to dashboard

**Expected Results:**
- ✓ Login form accepts credentials
- ✓ Success message appears
- ✓ Redirect to dashboard occurs
- ✓ Token is stored in localStorage
- ✓ User info displays in header

**API Call to Verify:**
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

### 5.2 Verify Protected Endpoints Require Auth

**Steps:**
1. Clear localStorage (to remove token)
2. Try to access dashboard directly
3. Observe redirect to login

**Expected Results:**
- ✓ Redirect to login page occurs
- ✓ Error message indicates authentication required
- ✓ After login, redirect back to originally requested page

### 5.3 Verify Token Expiration Handling

**Steps:**
1. Login successfully
2. Manually modify token in localStorage to invalid value
3. Try to access any protected page
4. Observe error handling

**Expected Results:**
- ✓ Error message indicates invalid/expired token
- ✓ Redirect to login page
- ✓ No application crash

### 5.4 Verify Logout Flow

**Steps:**
1. Click logout button
2. Observe redirect to login page
3. Try to access protected pages

**Expected Results:**
- ✓ Token is removed from localStorage
- ✓ Redirect to login page occurs
- ✓ Cannot access protected pages without re-login
- ✓ WebSocket connection is closed

---

## Automated Test Execution

Run the automated E2E tests:

```bash
# Run all E2E integration tests
pytest tests/integration/test_frontend_e2e_integration.py -v

# Run specific test class
pytest tests/integration/test_frontend_e2e_integration.py::TestDashboardIntegration -v

# Run with coverage
pytest tests/integration/test_frontend_e2e_integration.py --cov=src/api --cov-report=html
```

---

## Common Issues and Troubleshooting

### Issue: Dashboard stats show all zeros

**Solution:**
- Upload at least one model
- Start at least one optimization session
- Refresh the page

### Issue: WebSocket shows "Disconnected"

**Solution:**
- Verify backend server is running
- Check CORS configuration allows WebSocket connections
- Check browser console for WebSocket errors
- Verify Socket.IO is properly mounted in backend

### Issue: Authentication fails

**Solution:**
- Verify credentials are correct (admin/admin)
- Check backend logs for authentication errors
- Clear browser cache and localStorage
- Verify JWT token generation is working

### Issue: Sessions list is empty

**Solution:**
- Start at least one optimization session
- Check that MemoryManager is properly initialized
- Verify sessions are being persisted
- Check backend logs for errors

---

## Test Results Checklist

Use this checklist to track test completion:

### Dashboard Integration
- [ ] Statistics load correctly
- [ ] Error handling works when backend unavailable
- [ ] Data refreshes on page reload
- [ ] All statistics display correct data types
- [ ] Loading states work properly

### Sessions List Integration
- [ ] Sessions display correctly
- [ ] Status filtering works
- [ ] Date range filtering works
- [ ] Model ID filtering works
- [ ] Pagination works correctly
- [ ] Empty state displays properly
- [ ] Multiple filter combinations work

### Configuration Integration
- [ ] Configuration loads correctly
- [ ] Updates save properly
- [ ] Validation errors display correctly
- [ ] Various configuration values work
- [ ] Changes persist after refresh

### WebSocket Integration
- [ ] Connection status indicator works
- [ ] Progress updates appear in real-time
- [ ] Reconnection works after disconnect
- [ ] Multiple tabs receive updates
- [ ] No duplicate or missed updates

### Authentication Flows
- [ ] Login flow works
- [ ] Protected endpoints require auth
- [ ] Token expiration handled correctly
- [ ] Logout flow works
- [ ] Invalid credentials rejected

---

## Performance Benchmarks

Expected performance metrics:

- **Dashboard Load Time**: < 2 seconds
- **Sessions List Load Time**: < 1 second (for 100 sessions)
- **Configuration Load Time**: < 500ms
- **WebSocket Connection Time**: < 1 second
- **Real-time Update Latency**: < 2 seconds
- **API Response Time**: < 500ms (95th percentile)

---

## Next Steps

After completing all tests:

1. Document any issues found
2. Create bug reports for failures
3. Update this guide with any new findings
4. Run automated tests in CI/CD pipeline
5. Perform load testing with multiple concurrent users
