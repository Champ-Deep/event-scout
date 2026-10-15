// Writes the OpenUI system prompt for Ask AI to apps/backend/openui_prompt.txt.
// Run after changing src/features/ask/openui/spec.ts:   node scripts/gen-openui-prompt.mjs
// (Node 22.18+ runs the .ts spec directly.)
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { createLibrary, defineComponent, generateSystemPrompt } from '@openuidev/lang-core';
import { defineAskComponents, OPENUI_PROMPT_OPTIONS, OPENUI_ROOT } from '../src/features/ask/openui/spec.ts';

const here = dirname(fileURLToPath(import.meta.url));
const nullRenderers = new Proxy({}, { get: () => () => null });
const library = createLibrary({ components: defineAskComponents(defineComponent, nullRenderers), root: OPENUI_ROOT });
const prompt = generateSystemPrompt({ library: library.toSpec(), promptOptions: OPENUI_PROMPT_OPTIONS });

const out = resolve(here, '../../backend/openui_prompt.txt');
writeFileSync(out, prompt + '\n', 'utf8');
console.log(`Wrote ${out} (${prompt.length} chars)`);
