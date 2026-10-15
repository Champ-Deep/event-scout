# EventScout Mobile Application

## 📱 Overview
A React Native/Expo mobile application that integrates with all EventScout user-facing APIs. The app provides a seamless mobile experience for contact management, event networking, AI-powered insights, and offline sync capabilities.

## 🎯 Features

### **Core Screens** (Tab Navigation)

| Screen | Icon | API Integration | Description |
|--------|------|-----------------|-------------|
| **Contacts** | 👥 | `GET /api/contacts` <br> `GET /api/contacts/{id}` <br> `POST /api/contacts` <br> `PATCH /api/contacts/{id}` <br> `DELETE /api/contacts/{id}` | Manage all scanned contacts with search, filters, and scoring |
| **Scan** | 📷 | `POST /add_contact_from_image/` <br> `POST /scan_qr/` <br> `POST /add_contact/` | Scan business cards, QR codes, or manual entry |
| **Events** | 📅 | `GET /api/events` <br> `GET /api/events/{id}/seats` | View active events, seat rosters, and event stats |
| **AI Assistant** | ⚡ | `GET /api/models` <br> `POST /api/converse` <br> `GET /api/admin/activity` | AI-powered insights, event intelligence, and chat |
| **Profile** | 👤 | `GET /user/profile/` <br> `PUT /user/profile/` <br> `GET /dashboard` <br> `GET /export_contacts/` | User profile, settings, stats, and data export |

### **API Integration Status**

| API Category | Endpoints Integrated | Status |
|--------------|---------------------|--------|
| ✅ **Health & Status** | `GET /api/health` | Fully Integrated |
| ✅ **Contacts Management** | 5 endpoints | Fully Integrated |
| ✅ **Events & Seats** | 2 endpoints | Fully Integrated |
| ✅ **Sync Engine** | `POST /api/sync/push` <br> `GET /api/sync/pull` <br> `GET /api/sync/state` | Fully Integrated |
| ✅ **AI Copilot** | `GET /api/models` <br> `POST /api/converse` | Fully Integrated |
| ✅ **Exhibitors** | `GET /api/exhibitors` <br> `GET /api/exhibitors/categories` | Fully Integrated |
| ✅ **AI Scoring** | `POST /api/contact/{id}/score` <br> `POST /api/contacts/score_all` <br> `POST /api/contact/{id}/enrich` | Fully Integrated |
| ✅ **Authentication** | LinkedIn OAuth, Login/Register | Fully Integrated |
| ✅ **Legacy APIs** | All user endpoints | Fully Integrated |

## 🏗️ Architecture

### **1. API Service Layer** (`src/lib/api.ts`)
- Centralized Axios client with interceptors
- TypeScript interfaces for all API responses
- Error handling and retry logic
- Authentication token management

### **2. Reusable UI Components** (`src/ui/components.tsx`)
- **ContactCard**: Contact display with avatar and score badge
- **Button**: Multiple variants (primary, secondary, outline, danger)
- **IconButton**: Icon-only buttons with badges
- **ScoreBadge**: Visual score indicator (A-F)
- **EmptyState**: Placeholder for empty lists
- **Header**: Screen header with icons
- **TabBarIcon**: Tab navigation with badge indicators

### **3. Screens Design** (`src/features/AppNavigator.tsx`)
- **5-Tab Navigation**: Contacts, Scan, Events, Assistant, Profile
- **Offline-First**: Sync engine integration
- **Real-time Updates**: SSE live events
- **AI Integration**: Chat interface with Gemini/Claude

### **4. Navigation Structure**
```typescript
MainTabs
├── ContactsScreen (List, Search, Filters)
├── ScanScreen (Camera, OCR, QR, Manual)
├── EventsScreen (Events, Seats, Stats)
├── AssistantScreen (AI Chat, Insights)
└── ProfileScreen (User, Settings, Export)
```

## 🔧 Installation & Setup

### **1. Install Dependencies**
```bash
cd apps/mobile
npm install
# or
yarn install
```

### **2. Configure API Base URL**
Edit `src/lib/api.ts`:
```typescript
constructor(baseURL: string = 'https://your-eventscout-api.com') {
```

### **3. Run Development Server**
```bash
# iOS
npm run ios

# Android
npm run android

# Web
npm run web
```

### **4. Environment Variables**
Create `.env` file:
```env
EXPO_PUBLIC_API_BASE_URL=https://eventscout-backend.onrender.com
EXPO_PUBLIC_API_KEY=your_api_key_here
```

## 🎨 Design System

### **Colors**
```css
Primary: #0f172a (Dark Navy)
Secondary: #2563eb (Blue)
Success: #059669 (Green)
Warning: #d97706 (Yellow)
Danger: #ef4444 (Red)
Background: #f8fafc (Light Gray)
Text: #0f172a (Dark), #64748b (Gray)
```

### **Icons (Ionicons)**
- Contacts: `people`, `people-outline`
- Scan: `camera`, `camera-outline`
- Events: `calendar`, `calendar-outline`
- Assistant: `flash`, `flash-outline`
- Profile: `person`, `person-outline`

### **Typography**
- Headers: 18-24px, Weight 600
- Body: 16px, Weight 400
- Labels: 14px, Weight 500
- Captions: 12px, Weight 400

## 🔌 API Integration Details

### **Contacts Screen** → `/api/contacts`
```typescript
// List with filters
api.listContacts({
  search: 'John',
  stage: 'qualified',
  scoreBand: 'A',
  eventId: 'event-whx-2026',
  limit: 50
});

// Create contact
api.createContact({
  firstName: 'John',
  lastName: 'Doe',
  company: 'Acme Inc',
  email: 'john@acme.com',
  score: 85,
  scoreBand: 'A'
});
```

### **Sync Engine** → `/api/sync/*`
```typescript
// Push offline changes
api.syncPush({
  items: [/* outbox items */]
});

// Pull server updates
api.syncPull({
  cursor: 'last-sync-token',
  eventId: 'current-event'
});
```

### **AI Assistant** → `/api/converse`
```typescript
api.converseAI(
  "Show me my top leads from today",
  "claude-opus",
  conversationHistory
);
```

## 📱 Screens Overview

### **1. Contacts Screen**
- Search bar with real-time filtering
- Score-based filters (Tier A, B, C)
- Contact cards with avatars and scores
- Pull-to-refresh sync
- Floating "Add Contact" button

### **2. Scan Screen**
- Business card scanner (camera + OCR)
- QR code scanner for vCards
- Manual entry form
- Recent scans preview
- Quick action buttons

### **3. Events Screen**
- Active events list
- Event details and stats
- Seat roster viewing
- Event-specific contacts
- Join event functionality

### **4. AI Assistant Screen**
- Chat interface with AI
- Quick questions (Top Leads, Event Stats, etc.)
- Real-time responses
- Conversation history
- Multiple AI model support

### **5. Profile Screen**
- User profile and avatar
- Event statistics dashboard
- Quick actions (Sync, Export, Score All)
- Settings navigation
- Data export options

## 🚀 Deployment

### **Expo Build**
```bash
# Build for production
expo build:ios
expo build:android

# Build for EAS
eas build --platform all
```

### **Configuration**
```json
{
  "expo": {
    "name": "EventScout Mobile",
    "slug": "eventscout-mobile",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/icon.png",
    "splash": {
      "image": "./assets/splash.png"
    },
    "updates": {
      "fallbackToCacheTimeout": 0
    },
    "assetBundlePatterns": [
      "**/*"
    ]
  }
}
```

## 🔄 Offline Sync Strategy

### **Sync Flow**
1. **Local Storage**: SQLite database for offline use
2. **Outbox Queue**: Store pending changes locally
3. **Background Sync**: Push changes when online
4. **Conflict Resolution**: Version-based merge
5. **State Management**: Track sync status

### **Implementation** (`src/sync/`)
```typescript
interface SyncManager {
  pushChanges(): Promise<void>;
  pullUpdates(): Promise<void>;
  getSyncState(): SyncState;
  resolveConflicts(conflicts: Conflict[]): Promise<void>;
}
```

## 📊 Performance Optimizations

1. **Image Caching**: Avatar and QR code caching
2. **API Debouncing**: Search input debouncing
3. **Pagination**: Infinite scroll for contacts
4. **Lazy Loading**: Screen and component lazy loading
5. **Memory Management**: FlatList virtualization

## 🧪 Testing

### **Unit Tests**
```bash
npm test
```

### **Integration Tests**
- API service mocking
- Component testing
- Navigation testing
- State management testing

### **E2E Tests**
```bash
npm run test:e2e
```

## 📈 Future Enhancements

1. **Push Notifications**: Real-time event alerts
2. **Voice Notes**: Audio recording for contacts
3. **Location Services**: Booth mapping at events
4. **Calendar Integration**: Schedule meetings
5. **Advanced Analytics**: Contact engagement tracking
6. **Team Collaboration**: Shared contact lists
7. **Export Formats**: PDF reports, Excel exports

## 🛠️ Development Notes

### **Adding New API Endpoints**
1. Add interface to `src/lib/api.ts`
2. Add method to `EventScoutAPI` class
3. Update TypeScript types if needed
4. Test with mock data
5. Integrate into appropriate screen

### **Adding New Screens**
1. Create screen component in `src/features/`
2. Add to navigation stack
3. Implement API integration
4. Add to tab navigation with icon
5. Test navigation flow

### **Styling Guidelines**
- Use design system tokens
- Follow component composition
- Responsive design for all devices
- Accessibility (contrast, labels, focus)

## 📚 API Documentation Links

- [EventScout Backend API](https://github.com/your-org/eventscout-backend)
- [API Routes Documentation](https://eventscout-api.com/docs)
- [Legacy API Reference](https://eventscout-api.com/legacy-docs)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request
5. Review and merge

## 📄 License

MIT License - See LICENSE file for details

---

**EventScout Mobile** - Smart networking for modern events