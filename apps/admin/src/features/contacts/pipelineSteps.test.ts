import { describe, expect, it } from 'vitest';
import { PIPELINE_STEPS, stepStates } from './pipelineSteps';

const run = (status: string, running: boolean, currentStep?: string) => ({ status, running, ...(currentStep ? { currentStep } : {}) });

describe('stepStates', () => {
  it('has one state per step', () => {
    expect(stepStates(run('none', false))).toHaveLength(PIPELINE_STEPS.length);
  });
  it('running: done before the current step, now on it, waiting after', () => {
    expect(stepStates(run('pitching', true))).toEqual(['done', 'done', 'now', 'wait', 'wait']);
    expect(stepStates(run('researching', true))).toEqual(['now', 'wait', 'wait', 'wait', 'wait']);
  });
  it('running but not on a step yet (pending): all waiting', () => {
    expect(stepStates(run('pending', true))).toEqual(['wait', 'wait', 'wait', 'wait', 'wait']);
  });
  it('complete: every step done', () => {
    expect(stepStates(run('complete', false))).toEqual(['done', 'done', 'done', 'done', 'done']);
  });
  it('complete without a deck: deck and attach wait (shown as skipped)', () => {
    expect(stepStates(run('complete_no_deck', false))).toEqual(['done', 'done', 'done', 'wait', 'wait']);
  });
  it('failed: stops at the step named in currentStep, earlier ones done', () => {
    expect(stepStates(run('failed', false, 'Pitch: the AI timed out'))).toEqual(['done', 'done', 'failed', 'wait', 'wait']);
  });
  it('failed with no recognisable step: all waiting', () => {
    expect(stepStates(run('failed', false, 'Something broke'))).toEqual(['wait', 'wait', 'wait', 'wait', 'wait']);
    expect(stepStates(run('failed', false))).toEqual(['wait', 'wait', 'wait', 'wait', 'wait']);
  });
  it('not run: all waiting', () => {
    expect(stepStates(run('none', false))).toEqual(['wait', 'wait', 'wait', 'wait', 'wait']);
  });
});
