import { useBackend } from "../../../services/backend";
import { paths } from "../../../services/backend/endpoints";
import { SectionHeader } from "../../atoms/SectionHeader";
import { ImageCard } from "./ImageCard";
import { ProcessRadio } from "./ProcessRadio";
import { ProcessOptions } from "./ProcessRadio/ProcessOptions";
import { SearchBar } from "./SearchBar";
import { useQuery } from "@tanstack/react-query";
import Fuse from "fuse.js";
import { useState } from "react";

export function ImagesSection() {
  const { backend } = useBackend();
  const { data: images } = useQuery({
    queryKey: ["/images"],
    queryFn: () =>
      backend
        .get<
          paths["/images"]["get"]["responses"]["200"]["content"]["application/json"]
        >("/images")
        .then((_) => _.data),
  });

  const [search, setSearch] = useState("");
  const [option, setOption] = useState<ProcessOptions>(ProcessOptions.All);

  const imageDisplayed =
    images &&
    (search === ""
      ? images
      : new Fuse(images, { keys: ["name"] }).search(search).map((_) => _.item));
  return (
    <section>
      <SectionHeader name="Images" className="max-md:hidden md:mb-3" />
      <div className="mb-3 flex max-md:flex-col max-md:space-y-3 md:space-x-3">
        <SearchBar
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <ProcessRadio value={option} setValue={setOption} />
      </div>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {imageDisplayed?.map((image) => (
          <ImageCard key={image.id} image={image} />
        ))}
      </div>
    </section>
  );
}
