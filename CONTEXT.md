# EventScout Domain Model

## Core Concepts

**Event**: A conference, trade show, or gathering where contacts are made. Has a name, dates, timezone, location, and active status.

**Contact**: A person met at an event. Has personal details (name, company, title, contact info), a scoring status, and belongs to an event.

**Scoring**: The process of evaluating contacts using ICP (Ideal Customer Profile) criteria. Contacts receive a numeric score (0-100) and a letter band (A-F).

**Seat**: A user's access to an event. Defines their role (rep, admin, owner) and status.

**Sync**: State management for offline-capable mobile apps. Tracks synchronization cursor and conflict resolution.

## Relationships

- An **Event** contains many **Contacts**
- A **Contact** belongs to exactly one **Event**
- A **Seat** grants a user access to an **Event**
- **Scoring** evaluates **Contacts** against ICP criteria
- **Sync** manages offline state for mobile event participation

## Key Processes

1. **Contact Capture**: Adding contacts through scanning, manual entry, or import
2. **ICP Scoring**: Evaluating contacts against ideal customer profile criteria
3. **Pipeline Management**: Moving contacts through stages (new → contacted → qualified → converted)
4. **Event Administration**: Managing event details, seats, and access control
5. **Offline Sync**: Handling data synchronization for mobile field use