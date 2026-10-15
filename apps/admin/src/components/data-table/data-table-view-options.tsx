// shadcn/ui Data Table "View" menu: toggle which columns show. Extra items (density, reset) go in children.
import * as React from "react"
import type { Table } from "@tanstack/react-table"
import { Settings2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface DataTableViewOptionsProps<TData> {
  table: Table<TData>
  /** Column id → label; columns without one fall back to their id. */
  labels?: Record<string, string>
  children?: React.ReactNode
}

export function DataTableViewOptions<TData>({ table, labels = {}, children }: DataTableViewOptionsProps<TData>) {
  const hideable = table.getAllColumns().filter((c) => typeof c.accessorFn !== "undefined" || labels[c.id])
  const shown = hideable.filter((c) => c.getIsVisible()).length

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="hidden h-8 cursor-pointer rounded-lg bg-card font-medium text-body shadow-none lg:flex" aria-label="Table columns">
          <Settings2 />
          Columns
          {shown < hideable.length ? <span className="rounded bg-secondary px-1.5 text-[10px] font-semibold tabular-nums text-muted-foreground">{shown}/{hideable.length}</span> : null}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52 rounded-xl">
        <DropdownMenuLabel>Table columns</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {hideable.map((c) => (
          <DropdownMenuCheckboxItem key={c.id} className="cursor-pointer" checked={c.getIsVisible()} disabled={!c.getCanHide()}
            onCheckedChange={(v) => c.toggleVisibility(!!v)} onSelect={(e) => e.preventDefault()}>
            {labels[c.id] ?? c.id}
          </DropdownMenuCheckboxItem>
        ))}
        {children}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
