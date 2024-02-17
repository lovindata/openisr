import { useBackend } from "../../../services/backend";
import { paths } from "../../../services/backend/endpoints";
import { Header } from "../../atoms/Header";
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

  const imagesShown =
    images &&
    (search === ""
      ? images
      : new Fuse(images, { keys: ["name"] }).search(search).map((_) => _.item));
  return (
    <section className="md:w-[588px] lg:w-[888px]">
      <Header name="Images" className="max-md:hidden md:mb-3" />
      <div className="mb-3 flex max-md:flex-col max-md:space-y-3 md:space-x-3">
        <SearchBar
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <ProcessRadio value={option} setValue={setOption} />
      </div>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {imagesShown?.map((image) => (
          <ImageCard
            key={image.id}
            image={image}
            onLatestProcessQueryShow={(latestProcess) =>
              option === ProcessOptions.All ||
              (latestProcess === undefined &&
                option === ProcessOptions.Waiting) ||
              (latestProcess &&
                latestProcess.status.ended === undefined &&
                option === ProcessOptions.Running) ||
              (latestProcess?.status.ended !== undefined &&
                option === ProcessOptions.Terminated)
            }
          />
        ))}
      </div>
    </section>
  );
}
