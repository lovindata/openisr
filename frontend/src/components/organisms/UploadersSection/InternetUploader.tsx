import { BorderBox } from "../../atoms/BorderBox";
import { SvgIcon } from "../../atoms/SvgIcon";

export function InternetUploader() {
  return (
    <BorderBox dashed className="relative h-16 w-32">
      <div className="m-auto flex h-full w-min items-center space-x-3">
        <SvgIcon type="internet" className="h-8 w-8" />
        <label>Internet upload</label>
      </div>
    </BorderBox>
  );
}
