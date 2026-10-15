// Dropdown menu on shadcn/ui DropdownMenu: used for row actions and header menus.
import React from 'react';
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuShortcut, DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Icon, type IconName } from './icons';

export interface MenuItem {
  label: string;
  icon?: IconName;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  /** draws a divider above this item */
  separator?: boolean;
  hint?: string;
}

export function Menu({ trigger, items, align = 'end', label }: { trigger: React.ReactNode; items: MenuItem[]; align?: 'start' | 'end' | 'center'; label?: string }) {
  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>{trigger}</DropdownMenuTrigger>
      <DropdownMenuContent align={align} sideOffset={6} collisionPadding={8} aria-label={label} className="min-w-[210px] rounded-xl p-1.5" onClick={(e) => e.stopPropagation()}>
        {items.map((it, i) => (
          <React.Fragment key={`${it.label}-${i}`}>
            {it.separator ? <DropdownMenuSeparator /> : null}
            <DropdownMenuItem disabled={!!it.disabled} variant={it.danger ? 'destructive' : 'default'} onSelect={() => it.onSelect()} className="cursor-pointer rounded-lg px-2.5 py-2">
              {it.icon ? <Icon name={it.icon} size={15} /> : null}
              <span className="flex-1">{it.label}</span>
              {it.hint ? <DropdownMenuShortcut>{it.hint}</DropdownMenuShortcut> : null}
            </DropdownMenuItem>
          </React.Fragment>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
