import { ImagesSection } from "@/v2/features/images/components/organisms/ImagesSection";
import { UploadersSection } from "@/v2/features/images/components/organisms/UploadersSection";
import { HeroSection } from "@/v2/features/shared/components/organisms/HeroSection";
import { AppLayout } from "@/v2/features/shared/components/templates/AppLayout";
import { useBackend } from "@/v2/services/backend";
import { paths } from "@/v2/services/backend/endpoints";
import { useQuery } from "@tanstack/react-query";

export function AppPg() {
  const { backend } = useBackend();
  const { data: cards } = useQuery({
    queryKey: ["/query/v1/app/cards"],
    queryFn: () =>
      backend
        .get<
          paths["/query/v1/app/cards"]["get"]["responses"]["200"]["content"]["application/json"]
        >("/query/v1/app/cards")
        .then((_) => _.data),
    refetchInterval: (query) =>
      query.state.data &&
      query.state.data.some((_) => _.status.type === "Stoppable")
        ? 1000
        : false,
  });

  return (
    <AppLayout>
      <div className="flex flex-col items-center space-y-3 p-4">
        <div className="flex max-md:flex-col max-md:space-y-3 md:space-x-3">
          <HeroSection />
          <UploadersSection />
        </div>
        <ImagesSection cards={cards} />
      </div>
    </AppLayout>
  );
}
