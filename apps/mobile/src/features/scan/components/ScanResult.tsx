import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { Icon } from '../../../ui/Icon';
import type { Contact } from '@es/types';
import { theme, makeStyles } from '../../../ui/theme';
import { isHot } from '../../contacts/scoreLevel';
import { Card } from '../../../ui/Card';
import { Button } from '../../../ui/Button';

interface ScanResultProps {
  contact: Contact;
  onSave: () => Promise<void>;
  onEdit: () => void;
  onCancel: () => void;
  saving?: boolean;
}

export const ScanResult: React.FC<ScanResultProps> = ({
  contact,
  onSave,
  onEdit,
  onCancel,
  saving = false,
}) => {
  const formatField = (value: string | null | undefined) => {
    return value || 'Not detected';
  };

  const handleSave = async () => {
    try {
      await onSave();
      Alert.alert('Success', 'Contact saved successfully!');
    } catch (error) {
      Alert.alert('Error', 'Failed to save contact. Please try again.');
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Scan Result</Text>
        <TouchableOpacity onPress={onCancel} style={styles.closeButton}>
          <Icon name="X" size="lg" color={theme.colors.text.body} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content}>
        {/* Contact Preview */}
        <Card style={styles.contactCard}>
          <View style={styles.avatar}>
            <Icon name="User" size="xl" color={theme.colors.brand[500]} />
          </View>
          
          <Text style={styles.contactName}>
            {formatField(contact.firstName)} {formatField(contact.lastName)}
          </Text>
          
          {contact.title && (
            <Text style={styles.contactTitle}>{formatField(contact.title)}</Text>
          )}
          
          {contact.company && (
            <Text style={styles.contactCompany}>{formatField(contact.company)}</Text>
          )}
        </Card>

        {/* Detected Fields */}
        <View style={styles.fieldsSection}>
          <Text style={styles.sectionTitle}>Detected Information</Text>
          
          <View style={styles.fieldGrid}>
            <View style={styles.field}>
              <Icon name="Mail" size="md" color={theme.colors.text.body} />
              <Text style={styles.fieldLabel}>Email</Text>
              <Text style={styles.fieldValue} numberOfLines={1}>
                {formatField(contact.email)}
              </Text>
            </View>
            
            <View style={styles.field}>
              <Icon name="Phone" size="md" color={theme.colors.text.body} />
              <Text style={styles.fieldLabel}>Phone</Text>
              <Text style={styles.fieldValue} numberOfLines={1}>
                {formatField(contact.phone)}
              </Text>
            </View>
            
            <View style={styles.field}>
              <Icon name="Building" size="md" color={theme.colors.text.body} />
              <Text style={styles.fieldLabel}>Company</Text>
              <Text style={styles.fieldValue} numberOfLines={1}>
                {formatField(contact.company)}
              </Text>
            </View>
            
            <View style={styles.field}>
              <Icon name="Briefcase" size="md" color={theme.colors.text.body} />
              <Text style={styles.fieldLabel}>Title</Text>
              <Text style={styles.fieldValue} numberOfLines={1}>
                {formatField(contact.title)}
              </Text>
            </View>
          </View>
        </View>

        {/* Scoring Information */}
        <View style={styles.scoringSection}>
          <Text style={styles.sectionTitle}>Lead Score</Text>
          <View style={styles.scoreDisplay}>
            <View style={[styles.scoreCircle, 
              isHot(contact) && styles.scoreA,
              contact.scoreBand === 'B' && styles.scoreB,
              contact.scoreBand === 'C' && styles.scoreC,
              contact.scoreBand === 'D' && styles.scoreD,
            ]}>
              <Text style={styles.scoreText}>{contact.score}</Text>
            </View>
            <View style={styles.scoreInfo}>
              <Text style={styles.scoreBand}>Grade: {contact.scoreBand}</Text>
              <Text style={styles.scoreStage}>Stage: {contact.stage}</Text>
            </View>
          </View>
        </View>
      </ScrollView>

      {/* Action Buttons */}
      <View style={styles.actions}>
        <Button
          title="Edit Details"
          onPress={onEdit}
          variant="outline"
          icon="Edit"
          style={styles.editButton}
        />
        
        <Button
          title={saving ? "Saving..." : "Save Contact"}
          onPress={handleSave}
          loading={saving}
          icon="CheckCircle"
          style={styles.saveButton}
        />
      </View>
    </View>
  );
};

const styles = makeStyles(() => ({
  container: {
    flex: 1,
    backgroundColor: theme.colors.bg.app,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.bg.divider,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: theme.colors.text.display,
  },
  closeButton: {
    padding: 4,
  },
  content: {
    flex: 1,
    padding: 20,
  },
  contactCard: {
    alignItems: 'center',
    padding: 20,
    marginBottom: 20,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: theme.colors.brand[50],
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  contactName: {
    fontSize: 24,
    fontWeight: '600',
    color: theme.colors.text.display,
    textAlign: 'center',
    marginBottom: 4,
  },
  contactTitle: {
    fontSize: 16,
    color: theme.colors.text.body,
    textAlign: 'center',
    marginBottom: 4,
  },
  contactCompany: {
    fontSize: 14,
    color: theme.colors.text.muted,
    textAlign: 'center',
  },
  fieldsSection: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: theme.colors.text.display,
    marginBottom: 16,
  },
  fieldGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  field: {
    width: '48%',
    backgroundColor: theme.colors.bg.surface,
    borderRadius: theme.radii.md,
    padding: 16,
    marginBottom: 12,
    alignItems: 'center',
  },
  fieldLabel: {
    fontSize: 12,
    color: theme.colors.text.muted,
    marginTop: 8,
    marginBottom: 4,
  },
  fieldValue: {
    fontSize: 14,
    color: theme.colors.text.display,
    textAlign: 'center',
  },
  scoringSection: {
    marginBottom: 24,
  },
  scoreDisplay: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.bg.surface,
    borderRadius: theme.radii.lg,
    padding: 20,
  },
  scoreCircle: {
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  scoreA: {
    backgroundColor: theme.colors.semantic.successBg,
  },
  scoreB: {
    backgroundColor: theme.colors.semantic.warningBg,
  },
  scoreC: {
    backgroundColor: theme.colors.semantic.warningBg,
  },
  scoreD: {
    backgroundColor: theme.colors.semantic.dangerBg,
  },
  scoreText: {
    fontSize: 24,
    fontWeight: '700',
    color: theme.colors.text.display,
  },
  scoreInfo: {
    flex: 1,
  },
  scoreBand: {
    fontSize: 18,
    fontWeight: '600',
    color: theme.colors.text.display,
    marginBottom: 4,
  },
  scoreStage: {
    fontSize: 14,
    color: theme.colors.text.body,
  },
  actions: {
    flexDirection: 'row',
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: theme.colors.bg.divider,
  },
  editButton: {
    flex: 1,
    marginRight: 8,
  },
  saveButton: {
    flex: 2,
    marginLeft: 8,
  },
}));