import os

path = 'c:/Users/harsha.c/Desktop/EventScout/apps/backend/app.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('# profile_update: UserProfileUpdate,', 'profile_update: UserProfileUpdate,')
text = text.replace('# update_data = profile_update.model_dump(exclude_unset=True)', 'update_data = profile_update.model_dump(exclude_unset=True)')
text = text.replace('# if not update_data:', 'if not update_data:')
text = text.replace('#     raise HTTPException(status_code=400, detail="No fields provided for update")', '    raise HTTPException(status_code=400, detail="No fields provided for update")')

text = text.replace('# if profile:', 'if profile:')
text = text.replace('#     existing = profile.profile_data or {}', '    existing = profile.profile_data or {}')
text = text.replace('#     for key, value in update_data.items():', '    for key, value in update_data.items():')
text = text.replace('#         if value is not None:', '        if value is not None:')
text = text.replace('#             existing[key] = value', '            existing[key] = value')
text = text.replace('#     profile.profile_data = existing', '    profile.profile_data = existing')
text = text.replace('#     profile.updated_at = datetime.now(timezone.utc)', '    profile.updated_at = datetime.now(timezone.utc)')
text = text.replace('# else:', 'else:')
text = text.replace('#     profile = UserProfileDB(', '    profile = UserProfileDB(')
text = text.replace('#         user_id=uuid.UUID(user_id),', '        user_id=uuid.UUID(user_id),')
text = text.replace('#         profile_data={k: v for k, v in update_data.items() if v is not None},', '        profile_data={k: v for k, v in update_data.items() if v is not None},')
text = text.replace('#     )', '    )')
text = text.replace('#     session.add(profile)', '    session.add(profile)')
text = text.replace('# await session.commit()', 'await session.commit()')
text = text.replace('# return {"status": "success", "profile": profile.profile_data}', '')
text = text.replace('return {"status": "error", "message": "Endpoint temporarily disabled"}', 'return {"status": "success", "profile": profile.profile_data}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Done")
