// The AI pipeline's five steps and where a run is on them (pure: the drawer's Pipeline tab and the
// AI pipelines page draw from it).
import type { PipelineRun } from '../../lib/types';
import type { IconName } from '../../ui/icons';

export const PIPELINE_STEPS: { key: string; label: string; icon: IconName }[] = [
  { key: 'researching', label: 'Research', icon: 'search' }, { key: 'scoring', label: 'Score', icon: 'flame' }, { key: 'pitching', label: 'Pitch', icon: 'mail' },
  { key: 'generating_deck', label: 'Deck', icon: 'file' }, { key: 'attaching', label: 'Attach', icon: 'paperclip' },
];

export type StepState = 'done' | 'now' | 'failed' | 'wait';

/** Each step's state for a run: running → done up to the current one; failed → stops at the failed step; complete → all done (no deck: deck + attach wait). */
export function stepStates(p: Pick<PipelineRun, 'status' | 'running' | 'currentStep'>): StepState[] {
  const idx = PIPELINE_STEPS.findIndex((s) => s.key === p.status);
  return PIPELINE_STEPS.map((s, i) => {
    if (p.running) return i < idx ? 'done' : i === idx ? 'now' : 'wait';
    if (p.status === 'failed') {
      // the run stops at the step it failed on; earlier ones finished
      const at = idx >= 0 ? idx : PIPELINE_STEPS.findIndex((x) => (p.currentStep || '').toLowerCase().includes(x.label.toLowerCase()));
      return at < 0 ? 'wait' : i < at ? 'done' : i === at ? 'failed' : 'wait';
    }
    if (p.status.startsWith('complete')) return s.key === 'generating_deck' || s.key === 'attaching' ? (p.status === 'complete' ? 'done' : 'wait') : 'done';
    return 'wait';
  });
}
