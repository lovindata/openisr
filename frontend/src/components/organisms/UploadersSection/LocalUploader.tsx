import { BorderBox } from "../../atoms/BorderBox";
import { SvgIcon } from "../../atoms/SvgIcon";

export function LocalUploader() {
  return (
    <BorderBox dashed className="relative h-16 w-32">
      <input
        type="file"
        accept="image/*"
        multiple
        className="absolute inset-0 opacity-0"
      />
      <div className="m-auto flex h-full w-min items-center space-x-3">
        <SvgIcon type="folder" className="h-8 w-8" />
        <label>Local upload</label>
      </div>
    </BorderBox>
  );
}
