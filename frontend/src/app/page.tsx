import { HeroSection } from "../components/organisms/HeroSection";
import { UploadersSection } from "../components/organisms/UploadersSection";
import { AppLayout } from "../components/templates/AppLayout";

export default function Page() {
  return (
    <AppLayout>
      <div className="flex space-x-3 p-4">
        <HeroSection />
        <UploadersSection />
      </div>
    </AppLayout>
  );
}
