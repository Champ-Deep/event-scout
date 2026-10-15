import { describe, expect, it } from 'vitest';
import { bandOf, contactLevel, countAtLevels, levelCounts, levelFrom, levelKeyOf, levelOf, repLevelOf } from './levels';

describe('levelOf (the local rule)', () => {
  it('hot ≥ 85, warm 70–84, cool 50–69, cold < 50', () => {
    expect([100, 85, 84, 70, 69, 50, 49, 0].map(levelKeyOf)).toEqual(['hot', 'hot', 'warm', 'warm', 'cool', 'cool', 'cold', 'cold']);
  });
  it('not scored → null / U', () => {
    expect(levelOf(null)).toBeNull();
    expect(levelOf(undefined)).toBeNull();
    expect(bandOf(null)).toBe('U');
  });
  it('carries the band letter, label, colour and tile tint', () => {
    expect(levelOf(90)).toMatchObject({ band: 'A', label: 'Hot', color: 'var(--heat-hot)', tint: 'rose' });
    expect(levelOf(10)).toMatchObject({ band: 'D', label: 'Cold', tint: 'slate' });
  });
});

describe('server level first', () => {
  it('uses contact.level when the server sent one', () => {
    expect(contactLevel({ score: 90, level: 'cool' })?.key).toBe('cool');
    expect(levelFrom(40, 'hot')?.band).toBe('A');
  });
  it('falls back to the rule when it is missing, null or unknown', () => {
    expect(contactLevel({ score: 90 })?.key).toBe('hot');
    expect(contactLevel({ score: 72, level: null })?.key).toBe('warm');
    expect(contactLevel({ score: 55, level: 'lukewarm' })?.key).toBe('cool');
    expect(contactLevel({ score: null })).toBeNull();
  });
  it('the rep level reads repLevel / repScore', () => {
    expect(repLevelOf({ repScore: 30, repLevel: 'warm' })?.key).toBe('warm');
    expect(repLevelOf({ repScore: 86 })?.key).toBe('hot');
  });
});

describe('counts', () => {
  it('levelCounts takes scores or scored rows', () => {
    expect(levelCounts([99, 85, 84, 70, 69, 50, 49, 0, null, undefined])).toEqual({ all: 10, A: 2, B: 2, C: 2, D: 2, U: 2 });
    expect(levelCounts([{ score: 90, level: 'warm' }, { score: null }])).toEqual({ all: 2, A: 0, B: 1, C: 0, D: 0, U: 1 });
  });
  it('countAtLevels: hot + warm', () => {
    expect(countAtLevels([{ score: 90 }, { score: 70 }, { score: 69 }, { score: null }, { score: 10, level: 'hot' }], ['hot', 'warm'])).toBe(3);
  });
});
