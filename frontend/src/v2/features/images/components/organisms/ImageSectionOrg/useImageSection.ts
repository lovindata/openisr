import { ProcessRadioOptions } from "@/v2/features/processes/components/organisms/ProcessRadioOrg/ProcessRadioOptions";
import { components } from "@/v2/services/backend/endpoints";
import Fuse from "fuse.js";
import { useState } from "react";

export function useImageSection(
  cards: components["schemas"]["CardMod"][] | undefined
) {
  const [search, setSearch] = useState("");
  const [option, setOption] = useState<ProcessRadioOptions>(
    ProcessRadioOptions.All
  );

  let cardsShown =
    cards === undefined
      ? []
      : search === ""
        ? cards
        : new Fuse(cards, { keys: ["name"] }).search(search).map((_) => _.item);
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

  return { cardsShown, search, option, setSearch, setOption };
}
