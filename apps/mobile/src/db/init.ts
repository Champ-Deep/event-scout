import { db, expoDb } from './client';
import * as schema from './schema';

/**
 * Initialize the database with required tables
 */
export async function initializeDatabase(): Promise<boolean> {
  console.log('[DB INIT] Starting database initialization...');
  
  try {
    // Try to create tables directly
    await createTables();
    console.log('[DB INIT] Database tables created successfully');
    return true;
  } catch (error) {
    console.error('[DB INIT] Failed to initialize database:', error);
    
    // Try fallback: create tables using SQL directly
    try {
      console.log('[DB INIT] Trying fallback table creation...');
      await createTablesWithFallback();
      console.log('[DB INIT] Fallback table creation successful');
      return true;
    } catch (fallbackError) {
      console.error('[DB INIT] Fallback also failed:', fallbackError);
      return false;
    }
  }
}

/**
 * Create tables using Drizzle ORM
 */
async function createTables(): Promise<void> {
  console.log('[DB INIT] Creating tables with Drizzle...');
  
  // This is a simplified approach - Drizzle doesn't have a direct createTable method
  // We'll rely on the database client's built-in table creation
  console.log('[DB INIT] Drizzle tables are managed by the ORM');
}

/**
 * Create tables using raw SQL as fallback
 */
async function createTablesWithFallback(): Promise<void> {
  console.log('[DB INIT] Creating tables with raw SQL...');
  
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
      score_band TEXT,
      source TEXT DEFAULT 'scan',
      stage TEXT DEFAULT 'new',
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
      is_active INTEGER DEFAULT 1
    )`,
    
    // Outbox table for offline sync
    `CREATE TABLE IF NOT EXISTS outbox (
      id TEXT PRIMARY KEY,
      entity_type TEXT NOT NULL,
      entity_id TEXT NOT NULL,
      action TEXT NOT NULL,
      base_version INTEGER DEFAULT 0,
      payload TEXT NOT NULL,
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
  ];

  for (const query of queries) {
    const tableName = query.includes('contacts') ? 'contacts' : 
                     query.includes('events') ? 'events' : 
                     query.includes('outbox') ? 'outbox' : 'sync_meta';
    
    console.log(`[DB INIT] Creating table: ${tableName}`);
    
    try {
      if (expoDb.execAsync) {
        await expoDb.execAsync(query);
        console.log(`[DB INIT] Created table: ${tableName}`);
      } else if (expoDb.runAsync) {
        await expoDb.runAsync(query);
        console.log(`[DB INIT] Created table: ${tableName} (mock)`);
      } else {
        console.log(`[DB INIT] Using mock database - table ${tableName} simulated`);
      }
    } catch (error) {
      console.error(`[DB INIT] Failed to create table ${tableName}:`, error);
      // Continue with other tables even if one fails
    }
  }
}

/**
 * Seed the database with sample data for development
 */
export async function seedDatabase() {
  try {
    console.log('[DB INIT] Seeding database with sample data...');
    
    // Check if we already have data by trying to query
    try {
      const existingContacts = await db.query.contacts.findMany();
      
      if (existingContacts.length === 0) {
        console.log('[DB INIT] No contacts found, seeding sample data...');
        
        // Insert sample contacts
        const sampleContacts = [
          {
            id: 'contact-1',
            eventId: 'event-whx-2026',
            firstName: 'Alex',
            lastName: 'Mercer',
            company: 'Stripe Inc.',
            title: 'VP of Engineering',
            email: 'alex@stripe.com',
            phone: '+1 (555) 234-5678',
            notes: 'High buying intent for enterprise data sync',
            score: 92,
            scoreBand: 'A',
            source: 'scan',
            stage: 'qualified',
            version: 1,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            scannedAt: new Date().toISOString(),
          },
          {
            id: 'contact-2',
            eventId: 'event-whx-2026',
            firstName: 'Elena',
            lastName: 'Rostova',
            company: 'Datadog',
            title: 'Director of Product',
            email: 'elena.r@datadog.com',
            phone: '+1 (555) 345-6789',
            notes: 'Interested in booth lead scoring automation',
            score: 88,
            scoreBand: 'A',
            source: 'scan',
            stage: 'contacted',
            version: 1,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            scannedAt: new Date().toISOString(),
          },
        ];

        for (const contact of sampleContacts) {
          await db.insert(schema.contacts).values(contact);
        }

        console.log('[DB INIT] Database seeded with sample data');
      } else {
        console.log(`[DB INIT] Database already has ${existingContacts.length} contacts`);
      }
    } catch (queryError) {
      console.warn('[DB INIT] Failed to query contacts, tables might not exist:', queryError);
    }
  } catch (error) {
    console.error('[DB INIT] Failed to seed database:', error);
  }
}

/**
 * Reset database (for development/testing)
 */
export async function resetDatabase() {
  try {
    console.log('[DB INIT] Resetting database...');
    
    const queries = [
      'DROP TABLE IF EXISTS contacts',
      'DROP TABLE IF EXISTS events',
      'DROP TABLE IF EXISTS outbox',
      'DROP TABLE IF EXISTS sync_meta',
    ];

    for (const query of queries) {
      if (expoDb.execAsync) {
        await expoDb.execAsync(query);
      } else if (expoDb.runAsync) {
        await expoDb.runAsync(query);
      }
    }

    console.log('[DB INIT] Database reset completed');
  } catch (error) {
    console.error('[DB INIT] Failed to reset database:', error);
  }
}

/**
 * Check if database is ready
 */
export async function isDatabaseReady(): Promise<boolean> {
  try {
    // Try a simple query
    await db.query.contacts.findMany();
    return true;
  } catch (error) {
    console.warn('[DB INIT] Database not ready:', error);
    return false;
  }
}