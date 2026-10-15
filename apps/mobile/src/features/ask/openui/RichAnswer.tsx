import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View } from 'react-native';
import { Renderer } from '@openuidev/react-lang';
import { Text } from '../../../ui/Text';
import { theme, makeStyles } from '../../../ui/theme';
import { Markdown } from '../Markdown';
import { askLibrary, AskUiContext, type AskUiContextValue } from './library';

type Segment = { kind: 'text'; text: string } | { kind: 'ui'; code: string };

class ErrorBoundary extends React.Component<{ children: React.ReactNode, fallback: React.ReactNode, resetKey?: string }, { hasError: boolean }> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  // A half-written block can trip a component; try again when more of it arrives.
  override componentDidUpdate(prev: { resetKey?: string }) {
    if (this.state.hasError && prev.resetKey !== this.props.resetKey) this.setState({ hasError: false });
  }
  override render() {
    if (this.state.hasError) return this.props.fallback;
    return this.props.children;
  }
}

// ```openui-lang … ``` (also tolerates ```openui and a missing closing fence at the end)
const FENCE = /```(?:openui-lang|openui)[^\n]*\n([\s\S]*?)(?:```|$)/g;

/**
 * Splits an inline-mode answer into plain-text and OpenUI parts, in order. `streaming`: the answer
 * is still arriving, so a half-written opening fence at the end (` `` `, ` ```open `) isn't shown as text.
 */
export function splitAnswer(answer: string, streaming = false): Segment[] {
  let text = (answer || '').replace(/\r\n/g, '\n');
  if (streaming) text = text.replace(/(^|\n)`{1,3}[\w-]*$/, '$1');
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(FENCE)) {
    const before = text.slice(last, m.index).trim();
    if (before) out.push({ kind: 'text', text: before });
    if (m[1]!.trim()) out.push({ kind: 'ui', code: m[1]!.trim() });
    last = m.index! + m[0].length;
  }
  const rest = text.slice(last).trim();
  if (rest) out.push({ kind: 'text', text: rest });
  // A model that skipped the fences but wrote a program anyway.
  if (out.length === 1 && out[0]!.kind === 'text' && /^root\s*=\s*\w+\(/m.test(out[0]!.text)) {
    return [{ kind: 'ui', code: out[0]!.text }];
  }
  return out;
}

function UiBlock({ code, streaming }: { code: string; streaming: boolean }) {
  const [failed, setFailed] = useState(false);
  const parsedRoot = useRef(false);
  // While streaming the program is incomplete, so no root yet is normal: judge it once it's all here.
  useEffect(() => {
    if (!streaming && !parsedRoot.current) setFailed(true);
  }, [streaming]);
  if (failed) {
    return <Text style={styles.failed}>Part of this answer couldn't be displayed.</Text>;
  }
  return (
    <ErrorBoundary resetKey={streaming ? code : 'final'} fallback={streaming ? null : <Text style={styles.failed}>Part of this answer couldn't be displayed.</Text>}>
      <Renderer
        response={code}
        library={askLibrary}
        isStreaming={streaming}
        publishObservability={false}
        onParseResult={(r) => {
          parsedRoot.current = !!r?.root;
          if (!streaming && !r?.root) setFailed(true);
        }}
      />
    </ErrorBoundary>
  );
}

/**
 * An Ask AI answer: plain text (Markdown) plus native OpenUI components when the model sent
 * them. `contacts` are the ones the backend retrieved for this answer — ContactCards whose name
 * matches one of them open that contact. `streaming`: the answer is still arriving — cards build
 * up as their code comes in.
 */
export function RichAnswer({ text, contacts, onContact, streaming = false }: { text: string; streaming?: boolean } & AskUiContextValue) {
  const segments = useMemo(() => splitAnswer(text, streaming), [text, streaming]);
  const ctx = useMemo(() => ({ contacts, onContact }), [contacts, onContact]);
  return (
    <AskUiContext.Provider value={ctx}>
      <View style={styles.wrap}>
        {segments.map((s, i) =>
          s.kind === 'text' ? <Markdown key={i} text={s.text} /> : <UiBlock key={i} code={s.code} streaming={streaming} />,
        )}
      </View>
    </AskUiContext.Provider>
  );
}

/** True when the answer contains at least one OpenUI block (so callers can skip duplicate UI). */
export function hasUiBlock(text: string): boolean {
  return splitAnswer(text).some((s) => s.kind === 'ui');
}

const styles = makeStyles(() => ({
  wrap: { gap: 12 },
  failed: { fontSize: 12, color: theme.colors.text.muted, fontStyle: 'italic' },
}));
