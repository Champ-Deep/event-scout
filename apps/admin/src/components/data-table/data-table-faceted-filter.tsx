// shadcn/ui Data Table faceted filter: a dashed "+ Title" button opening a searchable checklist
// with per-option counts. Controlled, so the selection can live in the URL / on the server;
// `single` makes picking an option replace the previous one.
import * as React from "react"
import { Check, PlusCircle } from "lucide-react"

import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Separator } from "@/components/ui/separator"

export interface FacetOption {
  label: string
  value: string
  icon?: React.ReactNode
  count?: number
}

interface DataTableFacetedFilterProps {
  title: string
  options: FacetOption[]
  value: string[]
  onChange: (value: string[]) => void
  single?: boolean
}

export function DataTableFacetedFilter({ title, options, value, onChange, single }: DataTableFacetedFilterProps) {
  const [open, setOpen] = React.useState(false)
  const selected = new Set(value)
  const chosen = options.filter((o) => selected.has(o.value))

  const pick = (v: string) => {
    if (single) { onChange(selected.has(v) ? [] : [v]); setOpen(false); return }
    const next = new Set(selected)
    if (next.has(v)) next.delete(v); else next.add(v)
    onChange([...next])
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm"
          className={cn("h-8 cursor-pointer rounded-full border-[1.5px] bg-card font-bold shadow-none",
            selected.size > 0 ? "border-brand-300 bg-accent text-accent-foreground hover:bg-brand-100" : "border-dashed text-body")}>
          <PlusCircle />
          {title}
          {selected.size > 0 ? (
            <>
              <Separator orientation="vertical" className="mx-0.5 h-4" />
              <Badge variant="secondary" className="rounded-full bg-card px-1.5 font-bold lg:hidden">{selected.size}</Badge>
              <div className="hidden gap-1 lg:flex">
                {selected.size > 2 ? (
                  <Badge variant="secondary" className="rounded-full bg-card px-1.5 font-bold">{selected.size} selected</Badge>
                ) : (
                  chosen.map((o) => <Badge key={o.value} variant="secondary" className="max-w-40 truncate rounded-full bg-card px-1.5 font-bold">{o.label}</Badge>)
                )}
              </div>
            </>
          ) : null}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-60 overflow-hidden rounded-2xl p-0" align="start">
        <Command>
          <CommandInput placeholder={title} />
          <CommandList>
            <CommandEmpty>No results found.</CommandEmpty>
            <CommandGroup>
              {options.map((o) => {
                const on = selected.has(o.value)
                return (
                  <CommandItem key={o.value} value={`${o.label} ${o.value}`} onSelect={() => pick(o.value)} className="cursor-pointer">
                    <div className={cn("flex size-4 items-center justify-center border border-primary",
                      single ? "rounded-full" : "rounded-[4px]",
                      on ? "bg-primary text-primary-foreground" : "opacity-50 [&_svg]:invisible")}>
                      <Check className="size-3.5 text-primary-foreground" />
                    </div>
                    {o.icon}
                    <span className="truncate">{o.label}</span>
                    {o.count != null ? <span className="ml-auto flex h-4 min-w-4 items-center justify-center font-mono text-xs tabular-nums text-muted-foreground">{o.count}</span> : null}
                  </CommandItem>
                )
              })}
            </CommandGroup>
            {selected.size > 0 ? (
              <>
                <CommandSeparator />
                <CommandGroup>
                  <CommandItem onSelect={() => { onChange([]); setOpen(false) }} className="cursor-pointer justify-center text-center">Clear filter</CommandItem>
                </CommandGroup>
              </>
            ) : null}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
