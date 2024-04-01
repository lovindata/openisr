import { BorderBoxAtm } from "@/v2/features/shared/components/atoms/BorderBoxAtm";
import { SvgIconAtm } from "@/v2/features/shared/components/atoms/SvgIconAtm";

export function UploaderSectionInternet() {
  return (
    <BorderBoxAtm dashed className="h-16 w-32">
      <div className="mx-auto flex h-full w-min items-center space-x-3">
        <SvgIconAtm type="internet" className="h-8 w-8" />
        <label>Internet upload</label>
      </div>
    </BorderBoxAtm>
  );
}
