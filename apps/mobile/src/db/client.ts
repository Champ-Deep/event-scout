import { Platform } from 'react-native';
import * as schema from './schema';

let expoDb: any;
let db: any;
let isInitialized = false;

// Initialize database synchronously at module load time
function initializeDatabase() {
  if (isInitialized) return;
  
  try {
    if (Platform.OS !== 'web') {
      const { openDatabaseSync } = require('expo-sqlite');
      const { drizzle } = require('drizzle-orm/expo-sqlite');
      expoDb = openDatabaseSync('eventscout.db', { enableChangeListener: true });
      db = drizzle(expoDb, { schema });
      
      // Create tables if they don't exist
      createTablesSync();
      
      // Add transaction support for Drizzle
      db.transaction = async (callback: (tx: any) => Promise<void>) => {
        // For Expo SQLite, we need to use the native transaction API
        await expoDb.execAsync('BEGIN TRANSACTION');
        try {
          // Create a transaction context
          const txDb = { ...db };
          await callback(txDb);
          await expoDb.execAsync('COMMIT');
        } catch (error) {
          await expoDb.execAsync('ROLLBACK');
          throw error;
        }
      };
    } else {
      // Safe mock for Web preview with transaction support
      const mockContacts: any[] = [];
      const mockOutbox: any[] = [];
      
      expoDb = {
        execAsync: async () => {},
        runAsync: async () => ({ changes: 0, lastInsertRowId: 1 }),
      };
      
      db = {
        query: {
          contacts: {
            findMany: async () => mockContacts,
            findFirst: async (options: any) => {
              if (options?.where?.id?.eq) {
                return mockContacts.find(c => c.id === options.where.id.eq) || null;
              }
              return mockContacts[0] || null;
            },
          },
          outbox: {
            findMany: async () => mockOutbox,
          },
        },
        select: () => ({
          from: () => ({
            where: () => ({
              orderBy: () => ({
                limit: () => [],
              }),
            }),
          }),
        }),
        insert: (table: string) => ({
          values: (values: any) => ({
            returning: async () => {
              if (table === 'contacts') {
                const contact = Array.isArray(values) ? values[0] : values;
                mockContacts.push(contact);
                return [contact];
              } else if (table === 'outbox') {
                const outboxItem = Array.isArray(values) ? values[0] : values;
                mockOutbox.push(outboxItem);
                return [outboxItem];
              }
              return [];
            },
          }),
        }),
        // Add transaction mock for web
        transaction: async (callback: (tx: any) => Promise<void>) => {
          // Just execute the callback without actual transaction
          await callback(db);
        },
      };
    }
    
    isInitialized = true;
    console.log('Database initialized successfully');
  } catch (error) {
    console.error('Failed to initialize database:', error);
    throw error;
  }
}

function createTablesSync() {
  const queries = [
    // Contacts table
    `CREATE TABLE IF NOT EXISTS contacts (
      id TEXT PRIMARY KEY,
      event_id TEXT NOT NULL,
      first_name TEXT,
      last_name TEXT,
      company TEXT,
      title TEXT,
      email TEXT,
      phone TEXT,
      notes TEXT,
      score INTEGER,
      score_band TEXT CHECK(score_band IN ('A', 'B', 'C', 'D', 'F')),
      source TEXT DEFAULT 'scan' CHECK(source IN ('scan', 'manual', 'import')),
      stage TEXT DEFAULT 'new' CHECK(stage IN ('new', 'contacted', 'qualified', 'converted')),
      version INTEGER DEFAULT 1,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      scanned_at TEXT
    )`,
    
    // Events table
    `CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      start_date TEXT NOT NULL,
      end_date TEXT NOT NULL,
      timezone TEXT NOT NULL,
      location TEXT,
      is_active INTEGER DEFAULT 1 CHECK(is_active IN (0, 1))
    )`,
    
    // Outbox table for offline sync
    `CREATE TABLE IF NOT EXISTS outbox (
      id TEXT PRIMARY KEY,
      entity_type TEXT NOT NULL CHECK(entity_type IN ('contact', 'event')),
      entity_id TEXT NOT NULL,
      action TEXT NOT NULL CHECK(action IN ('create', 'update', 'delete')),
      base_version INTEGER DEFAULT 0,
      payload TEXT NOT NULL,
      created_at TEXT NOT NULL,
      attempts INTEGER DEFAULT 0,
      last_error TEXT
    )`,
    
    // Offline scan queue (photos waiting to be read by the server)
    `CREATE TABLE IF NOT EXISTS scan_queue (
      id TEXT PRIMARY KEY,
      kind TEXT NOT NULL,
      image_uri TEXT NOT NULL,
      event_name TEXT,
      created_at TEXT NOT NULL,
      attempts INTEGER DEFAULT 0,
      last_error TEXT
    )`,

    // Sync metadata table
    `CREATE TABLE IF NOT EXISTS sync_meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )`,
    
    // Create indexes for better performance
    `CREATE INDEX IF NOT EXISTS idx_contacts_event_id ON contacts(event_id)`,
    `CREATE INDEX IF NOT EXISTS idx_contacts_score_band ON contacts(score_band)`,
    `CREATE INDEX IF NOT EXISTS idx_contacts_stage ON contacts(stage)`,
    `CREATE INDEX IF NOT EXISTS idx_outbox_entity_type ON outbox(entity_type)`,
    `CREATE INDEX IF NOT EXISTS idx_outbox_created_at ON outbox(created_at)`,
  ];

  // Really synchronous (execSync): the async calls used to be fired without waiting, so the
  // first cache write / sync on a fresh install could run before the tables existed.
  const run = (query: string) => {
    if (expoDb.execSync) return expoDb.execSync(query);
    return expoDb.execAsync?.(query);
  };
  for (const query of queries) {
    try {
      run(query);
    } catch (error) {
      console.error(`Failed to create table: ${query.substring(0, 100)}`, error);
    }
  }

  // Migrations for installs created before these columns existed (ADD COLUMN fails if present).
  const migrations = [
    // Full contact as the server sent it (photo, LinkedIn, real source…), so offline views
    // show everything, not just the basic columns.
    `ALTER TABLE contacts ADD COLUMN raw TEXT`,
    // Event active when an offline scan was taken (sent as event_name; created_at = capture time).
    `ALTER TABLE scan_queue ADD COLUMN event_name TEXT`,
  ];
  for (const m of migrations) {
    try { run(m); } catch { /* column already there */ }
  }
}

// Initialize immediately
initializeDatabase();

export { db, expoDb };
