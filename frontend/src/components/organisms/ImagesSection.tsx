import { SectionHeader } from "../atoms/SectionHeader";
import { ImageCard } from "../molecules/ImageCard";
import { RadioProcess } from "../molecules/RadioProcess";
import { ProcessOptions } from "../molecules/RadioProcess/ProcessOptions";
import { SearchBar } from "../molecules/SearchBar";
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
    <div className="md:space-y-3">
      <SectionHeader name="Images" className="max-md:hidden" />
      <div className="flex max-md:flex-col max-md:space-y-3 md:space-x-3">
        <SearchBar
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <RadioProcess value={option} setValue={setOption} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        {images?.map((image) => (
          <ImageCard
            key={image.id}
            src={image.src}
            name={image.name}
            width={image.source.width}
            height={image.source.height}
          />
        ))}
      </div>
    </div>
  );
}
