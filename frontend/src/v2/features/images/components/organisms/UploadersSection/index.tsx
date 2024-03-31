import { InternetUploader } from "@/v2/features/images/components/organisms/UploadersSection/InternetUploader";
import { LocalUploader } from "@/v2/features/images/components/organisms/UploadersSection/LocalUploader";
import { Header } from "@/v2/features/shared/components/atoms/Header";

export function UploadersSection() {
  return (
    <section className="flex w-72 flex-col max-md:items-center md:space-y-3">
      <Header name="Uploaders" className="max-md:hidden" />
      <div className="flex space-x-3">
        <LocalUploader />
        <InternetUploader />
      </div>
    </section>
  );
}
