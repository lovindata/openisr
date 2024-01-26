import { BorderBox } from "../../atoms/BorderBox";
import { SvgIcon } from "../../atoms/SvgIcon";

interface Props {
  src: string;
  name: string;
  width: number;
  height: number;
}

export function ImageCard({ src, name, width, height }: Props) {
  return (
    <BorderBox className="flex h-20 w-72 items-center justify-between p-3 text-xs">
      <div className="w- flex space-x-3">
        <img src={src} alt={name} className="h-14 w-14 rounded-lg" />
        <div className="flex w-32 flex-col justify-between">
          <label className="truncate">{name}</label>
          <span>
            Source: {width}x{height}px
          </span>
          <span className="font-bold">Target: -</span>
        </div>
      </div>
      <div className="flex items-center space-x-3">
        <SvgIcon type="run" className="h-6 w-6 cursor-pointer" />
        <SvgIcon type="delete" className="h-6 w-6 cursor-pointer" />
      </div>
    </BorderBox>
  );
}
