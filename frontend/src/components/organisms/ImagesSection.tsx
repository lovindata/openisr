import { SectionHeader } from "../atoms/SectionHeader";
import { SearchBar } from "../molecules/SearchBar";
import { TabBar } from "../molecules/TabBar";
import { useState } from "react";

export function ImagesSection() {
  const [search, setSearch] = useState("");

  return (
    <div className="md:space-y-3">
      <SectionHeader name="Images" className="max-md:hidden" />
      <div className="flex max-md:flex-col max-md:space-y-3 md:space-x-3">
        <SearchBar
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <TabBar />
      </div>
    </div>
  );
}
