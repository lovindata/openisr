import { HeroSection } from "../components/organisms/HeroSection";
import { ImagesSection } from "../components/organisms/ImagesSection";
import { UploadersSection } from "../components/organisms/UploadersSection";
import { AppLayout } from "../components/templates/AppLayout";

export default function Page() {
  return (
    <AppLayout>
      <div className="flex flex-col items-center space-y-3 p-4">
        <div className="flex max-md:flex-col max-md:space-y-3 md:space-x-3">
          <HeroSection />
          <UploadersSection />
        </div>
        <ImagesSection />
      </div>
    </AppLayout>
  );
}
