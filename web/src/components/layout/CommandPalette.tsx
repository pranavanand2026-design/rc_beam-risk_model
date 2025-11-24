"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Command } from "cmdk";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { useBeamList } from "@/hooks/useFeatureMeta";
import { Search } from "lucide-react";

const MAX_VISIBLE = 50;

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const router = useRouter();
  const { data: beamList } = useBeamList();

  const filtered = useMemo(() => {
    if (!beamList) return [];
    const q = search.trim().toLowerCase();
    const matches = q
      ? beamList.beams.filter((b) => b.id.toLowerCase().includes(q))
      : beamList.beams;
    return matches.slice(0, MAX_VISIBLE);
  }, [beamList, search]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const handleSelect = useCallback(
    (beamId: string) => {
      setOpen(false);
      setSearch("");
      router.push(`/?beam=${encodeURIComponent(beamId)}`);
    },
    [router]
  );

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="p-0 overflow-hidden max-w-lg">
        <Command className="border-none" shouldFilter={false}>
          <div className="flex items-center border-b px-3">
            <Search className="mr-2 h-4 w-4 shrink-0 text-muted-foreground" />
            <Command.Input
              value={search}
              onValueChange={setSearch}
              placeholder="Search beams..."
              className="flex h-11 w-full bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground"
            />
            <kbd className="ml-2 inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
              ESC
            </kbd>
          </div>
          <Command.List className="max-h-72 overflow-y-auto p-2">
            <Command.Empty className="py-6 text-center text-sm text-muted-foreground">
              No beams found.
            </Command.Empty>
            {filtered.map((beam) => (
              <Command.Item
                key={beam.id}
                value={beam.id}
                onSelect={() => handleSelect(beam.id)}
                className="relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none data-[selected=true]:bg-accent data-[selected=true]:text-accent-foreground"
              >
                {beam.id}
              </Command.Item>
            ))}
            {filtered.length === MAX_VISIBLE && (
              <p className="py-2 text-center text-xs text-muted-foreground">
                Type to narrow results...
              </p>
            )}
          </Command.List>
        </Command>
      </DialogContent>
    </Dialog>
  );
}
