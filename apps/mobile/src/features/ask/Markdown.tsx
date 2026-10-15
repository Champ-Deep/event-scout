import React from 'react';
import { View, type TextStyle } from 'react-native';
import { Text } from '../../ui/Text';
import { theme, makeStyles } from '../../ui/theme';

/**
 * Minimal Markdown for assistant answers: #/##/### headings, - / * / 1. lists,
 * **bold**, *italic*, `code`, and paragraphs. Enough for the model's formatting
 * without pulling in a Markdown dependency.
 */

function inline(text: string, base: TextStyle, keyPrefix: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\*[^*\s][^*]*\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    const key = `${keyPrefix}-${i++}`;
    if (tok.startsWith('**') || tok.startsWith('__')) {
      out.push(<Text key={key} style={[base, styles.bold]}>{tok.slice(2, -2)}</Text>);
    } else if (tok.startsWith('`')) {
      out.push(<Text key={key} style={[base, styles.code]}>{tok.slice(1, -1)}</Text>);
    } else {
      out.push(<Text key={key} style={[base, styles.italic]}>{tok.slice(1, -1)}</Text>);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const blocks: React.ReactNode[] = [];
  let para: string[] = [];

  const flush = (key: string) => {
    if (!para.length) return;
    const body = para.join(' ');
    blocks.push(<Text key={key} style={styles.p}>{inline(body, styles.p, key)}</Text>);
    para = [];
  };

  lines.forEach((raw, idx) => {
    const line = raw.trimEnd();
    const key = `b${idx}`;
    const heading = /^(#{1,4})\s+(.*)$/.exec(line.trim());
    const bullet = /^(\s*)[-*•]\s+(.*)$/.exec(line);
    const numbered = /^(\s*)(\d+)[.)]\s+(.*)$/.exec(line);

    if (!line.trim()) { flush(key); return; }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) { flush(key); blocks.push(<View key={key} style={styles.rule} />); return; }
    if (heading) {
      flush(`${key}p`);
      const level = heading[1]!.length;
      const style = level <= 2 ? styles.h2 : styles.h3;
      blocks.push(<Text key={key} style={style}>{inline(heading[2]!.replace(/\*\*/g, ''), style, key)}</Text>);
      return;
    }
    if (bullet || numbered) {
      flush(`${key}p`);
      const indent = Math.min(2, Math.floor(((bullet?.[1] ?? numbered?.[1]) || '').length / 2));
      const marker = bullet ? '•' : `${numbered![2]}.`;
      const body = bullet ? bullet[2]! : numbered![3]!;
      blocks.push(
        <View key={key} style={[styles.li, { paddingLeft: indent * 14 }]}>
          <Text style={[styles.p, styles.marker]}>{marker}</Text>
          <Text style={[styles.p, styles.liText]}>{inline(body, styles.p, key)}</Text>
        </View>,
      );
      return;
    }
    para.push(line.trim());
  });
  flush('end');

  return <View style={styles.root}>{blocks}</View>;
}

const styles = makeStyles(() => ({
  root: { gap: 6 },
  p: { fontSize: 15, lineHeight: 22, color: theme.colors.text.display },
  bold: { fontWeight: '700' },
  italic: { fontStyle: 'italic' },
  code: { fontFamily: theme.typography.fontFamily.mono, backgroundColor: theme.colors.brand[50], color: theme.colors.brand[800], fontSize: 13 },
  h2: { fontSize: 17, lineHeight: 24, fontWeight: '800', color: theme.colors.text.display, marginTop: 6 },
  h3: { fontSize: 15, lineHeight: 22, fontWeight: '700', color: theme.colors.brand[800], marginTop: 4 },
  li: { flexDirection: 'row', gap: 8 },
  marker: { color: theme.colors.brand[600], fontWeight: '700', minWidth: 14 },
  liText: { flex: 1 },
  rule: { height: 1, backgroundColor: theme.colors.bg.divider, marginVertical: 6 },
}));
