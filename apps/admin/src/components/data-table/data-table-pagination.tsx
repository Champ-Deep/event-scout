// shadcn/ui Data Table footer: selected count, rows per page, page x of y, first / prev / next / last.
import * as React from "react"
import type { Table } from "@tanstack/react-table"
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface DataTablePaginationProps<TData> {
  table: Table<TData>
  pageSizes?: readonly number[]
  /** Extra text after the selection count (e.g. a row cap note). */
  note?: React.ReactNode
}

export function DataTablePagination<TData>({ table, pageSizes = [10, 20, 30, 40, 50], note }: DataTablePaginationProps<TData>) {
  const { pageIndex, pageSize } = table.getState().pagination
  const total = table.getFilteredRowModel().rows.length
  const pages = Math.max(1, table.getPageCount())
  const from = total ? pageIndex * pageSize + 1 : 0
  const to = Math.min(total, (pageIndex + 1) * pageSize)
  const picked = table.getFilteredSelectedRowModel().rows.length

  return (
    <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-1.5">
      <div className="min-w-0 flex-1 text-xs text-muted-foreground">
        Showing <b className="font-semibold tabular-nums text-foreground">{from.toLocaleString()}–{to.toLocaleString()}</b> of <b className="font-semibold tabular-nums text-foreground">{total.toLocaleString()}</b>
        {picked ? <span> · {picked.toLocaleString()} selected</span> : null}
        {note ? <span className="ml-1">{note}</span> : null}
      </div>
      <div className="flex items-center gap-x-3 sm:gap-x-4">
        <div className="flex items-center gap-1.5">
          <p className="hidden text-xs font-medium text-muted-foreground sm:block">Rows per page</p>
          <Select value={`${pageSize}`} onValueChange={(v) => table.setPageSize(Number(v))}>
            <SelectTrigger size="sm" className="!h-7 w-auto min-w-[58px] shrink-0 cursor-pointer gap-1 whitespace-nowrap rounded-full pl-2.5 pr-1.5 text-xs font-bold tabular-nums [&_svg]:size-3.5" aria-label="Rows per page"><SelectValue placeholder={pageSize} /></SelectTrigger>
            <SelectContent side="top">
              {pageSizes.map((n) => <SelectItem key={n} value={`${n}`}>{n}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center justify-center whitespace-nowrap text-xs font-bold tabular-nums sm:min-w-[72px]">
          <span className="hidden sm:inline">Page&nbsp;</span>{Math.min(pageIndex + 1, pages)} / {pages}
        </div>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="icon" className="hidden size-7 cursor-pointer rounded-full shadow-none lg:flex [&_svg]:size-3.5" onClick={() => table.setPageIndex(0)} disabled={!table.getCanPreviousPage()}>
            <span className="sr-only">Go to first page</span><ChevronsLeft />
          </Button>
          <Button variant="outline" size="icon" className="size-7 cursor-pointer rounded-full shadow-none [&_svg]:size-3.5" onClick={() => table.previousPage()} disabled={!table.getCanPreviousPage()}>
            <span className="sr-only">Go to previous page</span><ChevronLeft />
          </Button>
          <Button variant="outline" size="icon" className="size-7 cursor-pointer rounded-full shadow-none [&_svg]:size-3.5" onClick={() => table.nextPage()} disabled={!table.getCanNextPage()}>
            <span className="sr-only">Go to next page</span><ChevronRight />
          </Button>
          <Button variant="outline" size="icon" className="hidden size-7 cursor-pointer rounded-full shadow-none lg:flex [&_svg]:size-3.5" onClick={() => table.setPageIndex(table.getPageCount() - 1)} disabled={!table.getCanNextPage()}>
            <span className="sr-only">Go to last page</span><ChevronsRight />
          </Button>
        </div>
      </div>
    </div>
  )
}
