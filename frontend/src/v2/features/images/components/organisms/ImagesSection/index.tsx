import { ImageCard } from "@/v2/features/images/components/organisms/ImagesSection/ImageCard";
import { ProcessRadio } from "@/v2/features/processes/components/organisms/ProcessRadio";
import { ProcessOptions } from "@/v2/features/processes/components/organisms/ProcessRadio/ProcessOptions";
import { Header } from "@/v2/features/shared/components/atoms/Header";
import { SearchBar } from "@/v2/features/shared/components/molecules/SearchBar";
import { components } from "@/v2/services/backend/endpoints";
import Fuse from "fuse.js";
import { useState } from "react";

interface Props {
  cards: components["schemas"]["CardMod"][] | undefined;
}

export function ImagesSection({ cards }: Props) {
  const [search, setSearch] = useState("");
  const [option, setOption] = useState<ProcessOptions>(ProcessOptions.All);

  let cardsShown =
    cards &&
    (search === ""
      ? cards
      : new Fuse(cards, { keys: ["name"] }).search(search).map((_) => _.item));
  cardsShown =
    cardsShown &&
    cardsShown.filter((card) => {
      switch (option) {
        case ProcessOptions.All:
          return true;
        case ProcessOptions.Waiting:
          return card.status.type === "Runnable";
        case ProcessOptions.Running:
          return card.status.type === "Stoppable";
        case ProcessOptions.Terminated:
          return (
            card.status.type === "Errored" ||
            card.status.type === "Downloadable"
          );
      }
    });

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
        {cardsShown?.map((card) => (
          <ImageCard key={card.image_id} card={card} />
        ))}
      </div>
    </section>
  );
}
