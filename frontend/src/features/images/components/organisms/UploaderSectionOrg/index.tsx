import { UploaderSectionInternet } from "@/features/images/components/organisms/UploaderSectionOrg/UploaderSectionInternet";
import { UploaderSectionLocal } from "@/features/images/components/organisms/UploaderSectionOrg/UploaderSectionLocal";
import { HeaderAtm } from "@/features/shared/components/atoms/HeaderAtm";

export function UploaderSectionOrg() {
  return (
    <section className="flex w-72 flex-col max-md:items-center md:space-y-3">
      <HeaderAtm name="Uploaders" className="max-md:hidden" />
      <div className="flex space-x-3">
        <UploaderSectionLocal />
        <UploaderSectionInternet />
      </div>
    </section>
  );
}
