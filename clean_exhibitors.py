#!/usr/bin/env python3
"""
Clean exhibitor data: Remove speculative entries, fix booth errors, add verification fields.
"""

import json

# Read the current file
with open('whx_exhibitors.json', 'r') as f:
    data = json.load(f)

# Update event counts (WHX only, not including WHX Labs)
data['event']['total_exhibitors'] = '4300+'
data['event']['visitors'] = '235000+'
data['event']['note'] = 'Event counts are for WHX Dubai 2026 only (not including co-located WHX Labs Dubai)'

# Verified exhibitors (7) - with confirmed booth numbers from primary sources
verified_names = {
    'Siemens Healthineers',
    'Siora Surgicals Pvt. Ltd.',
    'iTD GmbH',
    'steute Meditec',
    'True Source Technology Co., Ltd.',
    'ICP DAS-BMP',
    'GWS Surgicals LLP'
}

# Confirmed exhibitors (11) - attendance verified, booth numbers estimated
confirmed_names = {
    'Emirates Health Services',
    'GE Healthcare',
    'United Imaging',
    'Philips',
    'Canon Medical Systems',
    'American Hospital Dubai',
    'Saudi German Health',
    'Almoosa Health Group',
    'Al Khayyat Investments',
    'Mediana',
    'Mindray'
}

# Filter to only keep verified and confirmed exhibitors
clean_exhibitors = []

for exhibitor in data['exhibitors']:
    name = exhibitor['name']

    # Skip all "realistic" (speculative) exhibitors
    if exhibitor.get('source') == 'realistic':
        print(f"REMOVED (speculative): {name}")
        continue

    # Fix booth number errors based on user verification
    if name == 'GWS Surgicals LLP':
        exhibitor['booth'] = 'N37.C58'  # Was H37.C58, official Swapcard shows N37.C58
        exhibitor['hall'] = 'North Hall N37'
        print(f"FIXED booth: {name} H37.C58 → N37.C58")

    if name == 'Mindray':
        exhibitor['booth'] = 'N21.D10'  # Was N23.B55, actual is N21.D10
        exhibitor['hall'] = 'North Hall N21'
        print(f"FIXED booth: {name} N23.B55 → N21.D10")

    if name == 'Mediana':
        exhibitor['booth'] = 'N27.B58'  # Was N23.B40, MedicalExpo shows N27.B58
        exhibitor['hall'] = 'North Hall N27'
        print(f"FIXED booth: {name} N23.B40 → N27.B58")

    # Add verification and booth_verified fields
    if name in verified_names:
        exhibitor['verification'] = 'verified'
        exhibitor['booth_verified'] = True
        exhibitor['source'] = 'verified'
        clean_exhibitors.append(exhibitor)
        print(f"KEPT (verified): {name}")
    elif name in confirmed_names or exhibitor.get('source') == 'confirmed':
        exhibitor['verification'] = 'confirmed'
        exhibitor['booth_verified'] = False  # Attendance confirmed, booth estimated
        exhibitor['source'] = 'confirmed'
        clean_exhibitors.append(exhibitor)
        print(f"KEPT (confirmed): {name}")
    else:
        # This should not happen if filtering works correctly
        print(f"WARNING: Unexpected exhibitor {name} with source {exhibitor.get('source')}")

# Update the data
data['exhibitors'] = clean_exhibitors

# Update data sources note
data['data_sources']['note'] = (
    "Only verified and confirmed exhibitors are included. "
    "Verified exhibitors have booth numbers confirmed from primary sources (company websites, press releases). "
    "Confirmed exhibitors have attendance verified but booth numbers are estimates. "
    "All speculative exhibitors have been removed to ensure data integrity."
)

# Save cleaned data
with open('whx_exhibitors_clean.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Clean data saved to whx_exhibitors_clean.json")
print(f"Total exhibitors: {len(clean_exhibitors)}")
print(f"Verified: {sum(1 for e in clean_exhibitors if e.get('verification') == 'verified')}")
print(f"Confirmed: {sum(1 for e in clean_exhibitors if e.get('verification') == 'confirmed')}")
