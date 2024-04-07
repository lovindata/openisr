import { ImageSectionCard } from "@/features/images/components/organisms/ImageSectionOrg/ImageSectionCard";
import { useImageSection } from "@/features/images/components/organisms/ImageSectionOrg/useImageSection";
import { ProcessRadioOrg } from "@/features/processes/components/organisms/ProcessRadioOrg";
import { HeaderAtm } from "@/features/shared/components/atoms/HeaderAtm";
import { SearchBarMol } from "@/features/shared/components/molecules/SearchBarMol";
import { components } from "@/services/backend/endpoints";

interface Props {
  cards: components["schemas"]["CardMod"][] | undefined;
}

export function ImageSectionOrg({ cards }: Props) {
  const { cardsShown, search, option, setSearch, setOption } =
    useImageSection(cards);

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
