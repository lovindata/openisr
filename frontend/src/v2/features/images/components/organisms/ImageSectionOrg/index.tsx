import { ImageSectionCard } from "@/v2/features/images/components/organisms/ImageSectionOrg/ImageSectionCard";
import { ProcessRadioOrg } from "@/v2/features/processes/components/organisms/ProcessRadioOrg";
import { ProcessRadioOptions } from "@/v2/features/processes/components/organisms/ProcessRadioOrg/ProcessRadioOptions";
import { HeaderAtm } from "@/v2/features/shared/components/atoms/HeaderAtm";
import { SearchBarMol } from "@/v2/features/shared/components/molecules/SearchBarMol";
import { components } from "@/v2/services/backend/endpoints";
import Fuse from "fuse.js";
import { useState } from "react";

interface Props {
  cards: components["schemas"]["CardMod"][] | undefined;
}

export function ImageSectionOrg({ cards }: Props) {
  const [search, setSearch] = useState("");
  const [option, setOption] = useState<ProcessRadioOptions>(
    ProcessRadioOptions.All
  );

  let cardsShown =
    cards &&
    (search === ""
      ? cards
      : new Fuse(cards, { keys: ["name"] }).search(search).map((_) => _.item));
  cardsShown =
    cardsShown &&
    cardsShown.filter((card) => {
      switch (option) {
        case ProcessRadioOptions.All:
          return true;
        case ProcessRadioOptions.Waiting:
          return card.status.type === "Runnable";
        case ProcessRadioOptions.Running:
          return card.status.type === "Stoppable";
        case ProcessRadioOptions.Terminated:
          return (
            card.status.type === "Errored" ||
            card.status.type === "Downloadable"
          );
      }
    });

  return (
    <section className="md:w-[588px] lg:w-[888px]">
      <HeaderAtm name="Images" className="max-md:hidden md:mb-3" />
      <div className="mb-3 flex max-md:flex-col max-md:space-y-3 md:space-x-3">
        <SearchBarMol
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <ProcessRadioOrg value={option} setValue={setOption} />
      </div>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {cardsShown?.map((card) => (
          <ImageSectionCard key={card.image_id} card={card} />
        ))}
      </div>
    </section>
  );
}
