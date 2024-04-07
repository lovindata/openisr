import { ImageSectionOrg } from "@/features/images/components/organisms/ImageSectionOrg";
import { UploaderSectionOrg } from "@/features/images/components/organisms/UploaderSectionOrg";
import { HeroSectionOrg } from "@/features/shared/components/organisms/HeroSectionOrg";
import { MainLayoutTpl } from "@/features/shared/components/templates/MainLayoutTpl";
import { useBackend } from "@/services/backend";
import { paths } from "@/services/backend/endpoints";
import { useQuery } from "@tanstack/react-query";

export function ImagePg() {
  const { backend } = useBackend();
  const { data: cards } = useQuery({
    queryKey: ["/queries/v1/app/cards"],
    queryFn: () =>
      backend
        .get<
          paths["/queries/v1/app/cards"]["get"]["responses"]["200"]["content"]["application/json"]
        >("/queries/v1/app/cards")
        .then((_) => _.data),
    refetchInterval: (query) =>
      query.state.data &&
      query.state.data.some((_) => _.status.type === "Stoppable")
        ? 1000
        : false,
  });

  return (
    <MainLayoutTpl>
      <div className="flex flex-col items-center space-y-3 p-4">
        <div className="flex max-md:flex-col max-md:space-y-3 md:space-x-3">
          <HeroSectionOrg />
          <UploaderSectionOrg />
        </div>
        <ImageSectionOrg cards={cards} />
      </div>
    </MainLayoutTpl>
  );
}
