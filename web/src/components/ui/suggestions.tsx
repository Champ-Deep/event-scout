// 21st.dev component id 12342 (serafimcloud/suggestions), source retrieved with the connector tool get_component.
// The shadcn CLI cannot install this registry item (its file entry has no type), so the source is committed here.
"use client";

import * as React from "react";
import type { ReactNode } from "react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type SuggestionItem = {
  id: string;
  label: string;
  value?: string;
  icon?: ReactNode;
  className?: string;
};

export type SuggestionsProps = {
  items: SuggestionItem[];
  onSelect: (item: SuggestionItem) => void;
  disabled?: boolean;
  className?: string;
  itemClassName?: string;
};

export function Suggestions({
  items,
  onSelect,
  disabled,
  className,
  itemClassName,
}: SuggestionsProps) {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          disabled={disabled}
          onClick={() => onSelect(item)}
          className={cn(
            "inline-flex h-7 items-center gap-1 rounded-[6px] border border-neutral-200 bg-transparent px-2 text-sm text-neutral-600 transition-colors hover:bg-neutral-100 hover:text-neutral-900 disabled:opacity-50 disabled:pointer-events-none dark:border-neutral-700 dark:text-neutral-400 dark:hover:bg-neutral-800 dark:hover:text-neutral-100",
            itemClassName,
            item.className,
          )}
        >
          {item.icon && (
            <span className="inline-flex shrink-0">{item.icon}</span>
          )}
          {item.label}
        </button>
      ))}
    </div>
  );
}
