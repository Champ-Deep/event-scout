// shadcn/ui Data Table column header: the title is a dropdown with Asc / Desc / Hide.
import * as React from "react"
import type { Column } from "@tanstack/react-table"
import { ArrowDown, ArrowUp, EyeOff } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface DataTableColumnHeaderProps<TData, TValue> extends React.HTMLAttributes<HTMLDivElement> {
  column: Column<TData, TValue>
  title: string
  icon?: React.ReactNode
  /** Labels for the two directions, e.g. "Newest first" / "Oldest first". */
  ascLabel?: string
  descLabel?: string
}

export function DataTableColumnHeader<TData, TValue>({
  column, title, icon, ascLabel = "Asc", descLabel = "Desc", className,
}: DataTableColumnHeaderProps<TData, TValue>) {
  const sorted = column.getIsSorted()
  if (!column.getCanSort() && !column.getCanHide()) {
    return <div className={cn("flex items-center gap-1.5", className)}>{icon}<span className="truncate">{title}</span></div>
  }

  return (
    <div className={cn("flex min-w-0 items-center", className)}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm"
            className={cn("-ml-2 h-8 max-w-full cursor-pointer gap-1.5 rounded-md px-2 font-mono text-[10.5px] font-medium uppercase tracking-[0.07em] data-[state=open]:bg-secondary",
              sorted ? "text-foreground" : "text-muted-foreground hover:text-foreground")}>
            {icon}
            <span className="truncate">{title}</span>
            {sorted === "desc" ? <ArrowDown className="size-3.5" /> : sorted === "asc" ? <ArrowUp className="size-3.5" /> : null}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          {column.getCanSort() ? (
            <>
              <DropdownMenuItem onClick={() => column.toggleSorting(false)} className={cn(sorted === "asc" && "font-medium text-primary")}>
                <ArrowUp /> {ascLabel}
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => column.toggleSorting(true)} className={cn(sorted === "desc" && "font-medium text-primary")}>
                <ArrowDown /> {descLabel}
              </DropdownMenuItem>
            </>
          ) : null}
          {column.getCanSort() && column.getCanHide() ? <DropdownMenuSeparator /> : null}
          {column.getCanHide() ? (
            <DropdownMenuItem onClick={() => column.toggleVisibility(false)}>
              <EyeOff /> Hide
            </DropdownMenuItem>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
