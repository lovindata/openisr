import { SectionHeader } from "../atoms/SectionHeader";
import { InternetUploader } from "../molecules/InternetUploader";
import { LocalUploader } from "../molecules/LocalUploader";

export function UploadersSection() {
  return (
    <div className="flex w-72 flex-col max-md:items-center md:space-y-3">
      <SectionHeader name="Uploaders" className="max-md:hidden" />
      <div className="flex space-x-3">
        <LocalUploader />
        <InternetUploader />
      </div>
    </div>
  );
}
