import React, { useState } from 'react';
import { View, TextInput, Pressable } from 'react-native';
import { Text } from '../../../ui/Text';
import { Icon } from '../../../ui/Icon';
import { theme, toneBg, makeStyles } from '../../../ui/theme';

export type AskComposerVariant = 'default' | 'compact' | 'offline' | 'half-sheet' | 'voice';

interface AskComposerProps {
  variant?: AskComposerVariant;
  scopeText?: string;
  placeholder?: string;
  onSend?: (text: string) => void;
  onVoiceToggle?: () => void;
  isListening?: boolean;
}

export function AskComposer({ 
  variant = 'default',
  scopeText = 'SCOPE',
  placeholder = 'Ask a follow-up',
  onSend,
  onVoiceToggle,
  isListening = false
}: AskComposerProps) {
  const [text, setText] = useState('');
  const isTyping = text.length > 0;
  
  const isCompact = variant === 'compact';
  const isHalfSheet = variant === 'half-sheet';
  const isOffline = variant === 'offline';

  if (isListening) {
    return (
      <View style={styles.listeningContainer}>
         <View style={styles.waveformRow}>
           <View style={[styles.waveBar, { height: 16, backgroundColor: toneBg('#cbd5e1') }]} />
           <View style={[styles.waveBar, { height: 24, backgroundColor: theme.colors.brand[600] }]} />
           <View style={[styles.waveBar, { height: 32, backgroundColor: theme.colors.fill.ink }]} />
           <View style={[styles.waveBar, { height: 40, backgroundColor: theme.colors.fill.brand800 }]} />
           <View style={[styles.waveBar, { height: 32, backgroundColor: theme.colors.brand[600] }]} />
           <View style={[styles.waveBar, { height: 24, backgroundColor: theme.colors.fill.brand700 }]} />
           <View style={[styles.waveBar, { height: 16, backgroundColor: toneBg('#94a3b8') }]} />
           <View style={[styles.waveBar, { height: 10, backgroundColor: toneBg('#cbd5e1'), borderRadius: 5 }]} />
           <View style={[styles.waveBar, { height: 6, backgroundColor: theme.colors.bg.divider, borderRadius: 3 }]} />
         </View>
         
         <View style={styles.voiceInputBox}>
            <View style={styles.voiceTextWrap}>
              <Text style={styles.voiceTextPrimary}>Listening...</Text>
              <Text style={styles.voiceTextSecondary}><View style={styles.voiceCursor} /></Text>
            </View>
            <View style={styles.micOuterRing}>
               <Pressable style={styles.micInnerBtn} onPress={onVoiceToggle}>
                  <Icon name="Mic" size="sm" color="white" />
                  <View style={styles.recordingDot} />
               </Pressable>
            </View>
         </View>

         <View style={styles.voiceInstructions}>
            <Text style={styles.instructionText}>← slide to cancel</Text>
            <Text style={styles.instructionText}>release to send →</Text>
         </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Scope Pill (Top Left) */}
      {!isCompact && !isHalfSheet && (
        <View style={styles.scopeRow}>
           <View style={styles.scopePill}>
             <Text style={styles.scopePillText}>{scopeText}</Text>
             <Icon name="ChevronDown" size="xs" color={theme.colors.text.display} />
           </View>
           {!isOffline && (
             <Text style={styles.contextItemsText}>Ready</Text>
           )}
        </View>
      )}

      {/* Main Input Row */}
      <View style={styles.inputRow}>
        {isCompact && !isOffline && (
          <View style={styles.compactLeftIcon}>
            <Icon name="Sparkle" size="sm" color={theme.colors.brand[700]} />
          </View>
        )}

        <View style={[
          styles.inputContainer,
          isOffline && styles.inputOffline,
          isTyping && !isOffline && styles.inputTyping
        ]}>
          {isHalfSheet && !isTyping && (
             <Icon name="Plus" size="sm" color={theme.colors.brand[700]} style={styles.inputLeftIcon} />
          )}
          <TextInput
            style={[styles.textInput, (isTyping && !isOffline) ? { color: theme.colors.text.display } : undefined]}
            placeholder={placeholder}
            placeholderTextColor={theme.colors.text.muted}
            value={text}
            onChangeText={setText}
            returnKeyType="send"
            onSubmitEditing={() => { if (text.trim()) { onSend?.(text); setText(''); } }}
          />
          {!isTyping && !isCompact && !isHalfSheet && !isOffline && (
             <View style={styles.inputRightIcons}>
                <Icon name="Paperclip" size="sm" color={theme.colors.text.muted} style={styles.iconMargin} />
             </View>
          )}
          {isHalfSheet && !isTyping && (
             <Icon name="Sparkles" size="sm" color={theme.colors.text.muted} style={styles.iconMarginRight} />
          )}
          {isCompact && !isTyping && (
             <Icon name="CornerDownLeft" size="xs" color={theme.colors.text.muted} style={styles.iconMarginRight} />
          )}
        </View>

        {/* Right Action Button */}
        <Pressable 
          style={[
             styles.actionBtn,
             isTyping && styles.actionBtnActive,
             (isOffline || isCompact) && !isTyping && styles.actionBtnOffline
          ]}
          onPress={() => {
            if (isTyping) { onSend?.(text); setText(''); } // clear after sending
            else onVoiceToggle?.();
          }}
        >
          {isTyping || isHalfSheet ? (
             <Icon name="ArrowUp" size="sm" color="white" />
          ) : isOffline || isCompact ? (
             <View style={styles.concentricCircle}>
               <View style={styles.innerDot} />
             </View>
          ) : (
             <Icon name="Mic" size="sm" color={theme.colors.text.display} />
          )}
        </Pressable>
      </View>
    </View>
  );
}

const styles = makeStyles(() => ({
  container: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 24, // Safe area pad
    backgroundColor: toneBg('#ffffff'),
    borderTopWidth: 1,
    borderTopColor: theme.colors.bg.divider,
  },
  scopeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  scopePill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: toneBg('#f1f5f9'),
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    gap: 4,
  },
  scopePillText: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 1,
    color: theme.colors.text.display,
  },
  contextItemsText: {
    fontSize: 12,
    color: theme.colors.text.muted,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  inputContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: toneBg('#ffffff'),
    borderWidth: 1,
    borderColor: theme.colors.bg.divider,
    borderRadius: 24,
    height: 48,
    paddingHorizontal: 16,
  },
  inputOffline: {
    borderWidth: 2,
    borderColor: theme.colors.text.display,
  },
  inputTyping: {
    borderWidth: 2,
    borderColor: theme.colors.fill.brand700,
  },
  inputLeftIcon: {
    marginRight: 8,
  },
  textInput: {
    flex: 1,
    fontSize: 15,
    fontFamily: theme.typography.fontFamily.sans,
    color: theme.colors.text.display,
  },
  inputRightIcons: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconMargin: {
    marginHorizontal: 8,
  },
  iconMarginRight: {
    marginLeft: 8,
  },
  compactLeftIcon: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  actionBtn: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: toneBg('#f1f5f9'),
    justifyContent: 'center',
    alignItems: 'center',
  },
  actionBtnActive: {
    backgroundColor: theme.colors.fill.brand700,
  },
  actionBtnOffline: {
    backgroundColor: toneBg('#ffffff'),
    borderWidth: 2,
    borderColor: theme.colors.text.display,
  },
  concentricCircle: {
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: theme.colors.text.display,
    justifyContent: 'center',
    alignItems: 'center',
  },
  innerDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: theme.colors.fill.ink,
  },
  /* Voice Listening Styles */
  listeningContainer: {
    paddingHorizontal: 20,
    paddingTop: 30,
    paddingBottom: 40,
    backgroundColor: toneBg('#ffffff'),
    borderTopWidth: 1,
    borderTopColor: theme.colors.bg.divider,
    alignItems: 'center',
  },
  waveformRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 32,
  },
  waveBar: {
    width: 6,
    borderRadius: 3,
  },
  voiceInputBox: {
    flexDirection: 'row',
    alignItems: 'center',
    width: '100%',
    borderWidth: 2,
    borderColor: theme.colors.text.display,
    borderRadius: 24,
    paddingLeft: 20,
    paddingVertical: 12,
    paddingRight: 8,
    marginBottom: 20,
  },
  voiceTextWrap: {
    flex: 1,
  },
  voiceTextPrimary: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.text.display,
    marginBottom: 4,
  },
  voiceTextSecondary: {
    fontSize: 16,
    color: theme.colors.text.muted,
  },
  voiceCursor: {
    width: 8,
    height: 16,
    backgroundColor: theme.colors.fill.brand700,
    marginLeft: 4,
  },
  micOuterRing: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: theme.colors.brand[100],
    justifyContent: 'center',
    alignItems: 'center',
    position: 'absolute',
    right: -10,
  },
  micInnerBtn: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: theme.colors.fill.ink,
    justifyContent: 'center',
    alignItems: 'center',
  },
  recordingDot: {
    position: 'absolute',
    top: 10,
    right: 12,
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.semantic.danger,
  },
  voiceInstructions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    paddingHorizontal: 10,
  },
  instructionText: {
    fontSize: 12,
    fontFamily: theme.typography.fontFamily.mono,
    color: theme.colors.text.muted,
    letterSpacing: 1,
  }
}));
