import { HeroSection } from "../components/organisms/HeroSection";
import { ImagesSection } from "../components/organisms/ImagesSection";
import { UploadersSection } from "../components/organisms/UploadersSection";
import { AppLayout } from "../components/templates/AppLayout";
import { useBackend } from "../services/backend";
import { paths } from "../services/backend/endpoints";
import { useQuery } from "@tanstack/react-query";

export default function Page() {
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

  return (
    <AppLayout>
      <div className="flex flex-col items-center space-y-3 p-4">
        <div className="flex max-md:flex-col max-md:space-y-3 md:space-x-3">
          <HeroSection />
          <UploadersSection />
        </div>
        <ImagesSection images={images} />
      </div>
    </AppLayout>
  );
}
