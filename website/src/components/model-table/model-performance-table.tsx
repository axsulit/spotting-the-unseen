"use client"

import * as React from "react"
import {
  IconChevronDown,
  IconLayoutColumns,
} from "@tabler/icons-react"
import {
  ColumnDef,
  ColumnFiltersState,
  flexRender,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
  VisibilityState,
} from "@tanstack/react-table"
import { z } from "zod"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Label } from "@/components/ui/label"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tabs,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import { useMemo } from "react"
import performanceData from "@/lib/performance.json"

export const dataSchema = z.object({
  id: z.string(),
  datasetName: z.string(),
  alterationType: z.string(),
  variation: z.string(),
  model: z.string(),
  accuracy: z.number(),
  f1: z.number(),
  precision: z.number(),
  recall: z.number(),
})

const columns: ColumnDef<z.infer<typeof dataSchema>>[] = [
  {
    id: "datasetName",
    header: "Dataset",
    cell: ({ row }) => {
      return <span>{row.original.datasetName}</span>
    },
    
    enableHiding: false,
  },
  {
    id: "alterationType",
    header: "Alteration Type",
    cell: ({ row }) => {
      return <span>{row.original.alterationType}</span>
    },
    enableSorting: false,
    enableHiding: false,
  },
  {
    accessorKey: "variation",
    header: "Variation",
    cell: ({ row }) => {
      return <span>{row.original.variation}</span>
    },
    enableHiding: false,
  },
  {
    accessorKey: "model",
    header: "Model",
    cell: ({ row }) => {
      return <span>{row.original.model}</span>
    },
  },
  {
    accessorKey: "accuracy",
    header: "Accuracy (%)",
    cell: ({ row }) => {
      return <span>{row.original.accuracy}</span>
    },
  },
  {
    accessorKey: "Precision",
    header: "Precision (%)",
    cell: ({ row }) => {
      return <span>{row.original.precision}</span>
    },
  },
  {
    accessorKey: "Recall",
    header: "Recall (%)",
    cell: ({ row }) => {
      return <span>{row.original.recall}</span>
    },
  },
  {
    accessorKey: "F1-Score",
    header: "F1-Score (%)",
    cell: ({ row }) => {
      return <span>{row.original.f1}</span>
    },
  },
]

  export function ModelPerformanceTable() {

  const initialData: z.infer<typeof dataSchema>[] = useMemo(() => {
    // Validate and map performanceData to match the schema
    return performanceData
      .map((item: z.infer<typeof dataSchema>) => {
        const parsed = dataSchema.safeParse(item)
        if (parsed.success) {
          return parsed.data
        }
        return null
      })
      .filter(Boolean) as z.infer<typeof dataSchema>[]
  }, [])

  const [columnVisibility, setColumnVisibility] =
    React.useState<VisibilityState>({})
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>(
    []
  )
  const [sorting, setSorting] = React.useState<SortingState>([])
  const [selectedDataset, setSelectedDataset] = React.useState<string>("all")

  // Get unique dataset names for the dropdown
  const datasetNames = useMemo(() => {
    const names = [...new Set(initialData.map(item => item.datasetName))]
    return names
  }, [initialData])

  // Filter data based on selected dataset
  const filteredData = useMemo(() => {
    if (selectedDataset === "all") {
      return initialData
    }
    return initialData.filter(item => item.datasetName === selectedDataset)
  }, [selectedDataset, initialData])

  const table = useReactTable({
    data: filteredData,
    columns,
    state: {
      sorting,
      columnVisibility,
      columnFilters,
    },
    getRowId: (row) => row.id,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
  })

  return (
    <Tabs
      defaultValue="outline"
      className="w-full flex-col justify-start gap-6"
    >
      <div className="flex items-center justify-between px-4 lg:px-6">
        <Label htmlFor="view-selector" className="sr-only">
          View
        </Label>
        <TabsList className="**:data-[slot=badge]:bg-muted-foreground/30 **:data-[slot=badge]:size-5 **:data-[slot=badge]:rounded-full **:data-[slot=badge]:px-1 flex">
          <TabsTrigger value="all" onClick={() => setSelectedDataset("all")}>
            All Datasets
          </TabsTrigger>
          {datasetNames.map((datasetName) => (
            <TabsTrigger key={datasetName} value={datasetName} onClick={() => setSelectedDataset(datasetName)}>
              {datasetName}
            </TabsTrigger>
          ))}
        </TabsList>
        <div className="flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <IconLayoutColumns />
                <span className="hidden lg:inline">Customize Columns</span>
                <span className="lg:hidden">Columns</span>
                <IconChevronDown />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              {table
                .getAllColumns()
                .filter(
                  (column) =>
                    typeof column.accessorFn !== "undefined" &&
                    column.getCanHide()
                )
                .map((column) => {
                  return (
                    <DropdownMenuCheckboxItem
                      key={column.id}
                      className="capitalize"
                      checked={column.getIsVisible()}
                      onCheckedChange={(value) =>
                        column.toggleVisibility(!!value)
                      }
                    >
                      {column.id}
                    </DropdownMenuCheckboxItem>
                  )
                })}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      <div
        className="relative flex flex-col gap-4 overflow-auto px-4 lg:px-6"
      >
        <div className="overflow-hidden rounded-lg border">
            <Table>
              <TableHeader className="bg-muted sticky top-0 z-10">
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => {
                      return (
                        <TableHead key={header.id} colSpan={header.colSpan}>
                          {header.isPlaceholder
                            ? null
                            : flexRender(
                                header.column.columnDef.header,
                                header.getContext()
                              )}
                        </TableHead>
                      )
                    })}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody className="**:data-[slot=table-cell]:first:w-8">
                {table.getRowModel().rows?.length ? (
                  (() => {
                    // Prepare merged row logic
                    const rows = table.getRowModel().rows;
                    let prevDataset: string | null = null, prevAlteration: string | null = null, prevVariation: string | null = null;
                    let datasetStart = 0, alterationStart = 0, variationStart = 0;
                    // Calculate rowSpans for each group
                    const datasetGroups: Record<string, {start: number, span: number}> = {};
                    const alterationGroups: Record<string, {start: number, span: number}> = {};
                    const variationGroups: Record<string, {start: number, span: number}> = {};
                    rows.forEach((row, i) => {
                      const d = row.original.datasetName;
                      const a = row.original.alterationType;
                      const v = row.original.variation;
                      // Dataset grouping
                      if (d !== prevDataset) {
                        if (prevDataset !== null) {
                          datasetGroups[`${prevDataset}-${datasetStart}`].span = i - datasetStart;
                        }
                        datasetGroups[`${d}-${i}`] = {start: i, span: 1};
                        datasetStart = i;
                      }
                      // Alteration grouping
                      if (a !== prevAlteration || d !== prevDataset) {
                        if (prevAlteration !== null) {
                          alterationGroups[`${prevDataset}-${prevAlteration}-${alterationStart}`].span = i - alterationStart;
                        }
                        alterationGroups[`${d}-${a}-${i}`] = {start: i, span: 1};
                        alterationStart = i;
                      }
                      // Variation grouping
                      if (v !== prevVariation || a !== prevAlteration || d !== prevDataset) {
                        if (prevVariation !== null) {
                          variationGroups[`${prevDataset}-${prevAlteration}-${prevVariation}-${variationStart}`].span = i - variationStart;
                        }
                        variationGroups[`${d}-${a}-${v}-${i}`] = {start: i, span: 1};
                        variationStart = i;
                      }
                      prevDataset = d;
                      prevAlteration = a;
                      prevVariation = v;
                    });
                    // Finalize last group
                    if (rows.length) {
                      datasetGroups[`${prevDataset}-${datasetStart}`].span = rows.length - datasetStart;
                      alterationGroups[`${prevDataset}-${prevAlteration}-${alterationStart}`].span = rows.length - alterationStart;
                      variationGroups[`${prevDataset}-${prevAlteration}-${prevVariation}-${variationStart}`].span = rows.length - variationStart;
                    }
                    // Render rows with merged cells
                    return rows.map((row, i) => {
                      const d = row.original.datasetName;
                      const a = row.original.alterationType;
                      const v = row.original.variation;
                      return (
                        <TableRow key={row.id} className="hover:bg-muted">
                          {/* Dataset Name */}
                          {datasetGroups[`${d}-${i}`] && datasetGroups[`${d}-${i}`].start === i ? (
                            <TableCell rowSpan={datasetGroups[`${d}-${i}`].span}>
                              {d}
                            </TableCell>
                          ) : null}
                          {/* Alteration Type */}
                          {alterationGroups[`${d}-${a}-${i}`] && alterationGroups[`${d}-${a}-${i}`].start === i ? (
                            <TableCell rowSpan={alterationGroups[`${d}-${a}-${i}`].span}>
                              {a}
                            </TableCell>
                          ) : null}
                          {/* Variation */}
                          {variationGroups[`${d}-${a}-${v}-${i}`] && variationGroups[`${d}-${a}-${v}-${i}`].start === i ? (
                            <TableCell rowSpan={variationGroups[`${d}-${a}-${v}-${i}`].span}>
                              {v}
                            </TableCell>
                          ) : null}
                          {/* Model, Accuracy, Precision, Recall, F1 */}
                          {row.getVisibleCells().slice(3).map((cell) => (
                            <TableCell key={cell.id}>
                              {flexRender(cell.column.columnDef.cell, cell.getContext())}
                            </TableCell>
                          ))}
                        </TableRow>
                      );
                    });
                  })()
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={columns.length}
                      className="h-24 text-center"
                    >
                      No results.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
        </div>
      </div>
    </Tabs>
  )
}

