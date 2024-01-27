import { SectionHeader } from "../../atoms/SectionHeader";
import { ImageCard } from "./ImageCard";
import { RadioProcess } from "./RadioProcess";
import { ProcessOptions } from "./RadioProcess/ProcessOptions";
import { SearchBar } from "./SearchBar";
import { useState } from "react";

interface Props {
  images:
    | {
        id: number;
        src: string;
        name: string;
        source: { width: number; height: number };
      }[]
    | undefined;
}

export function ImagesSection({ images }: Props) {
  const [search, setSearch] = useState("");
  const [option, setOption] = useState<ProcessOptions>(ProcessOptions.All);

  return (
    <div>
      <SectionHeader name="Images" className="max-md:hidden md:mb-3" />
      <div className="mb-3 flex max-md:flex-col max-md:space-y-3 md:space-x-3">
        <SearchBar
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <RadioProcess value={option} setValue={setOption} />
      </div>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {images?.map((image) => (
          <ImageCard
            key={image.id}
            id={image.id}
            src={image.src}
            name={image.name}
            source={image.source}
          />
        ))}
      </div>
    </div>
  );
}
